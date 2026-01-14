import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import threading
from copy import deepcopy

from ml_model.config import *
from ml_model.model import make_model
from ml_model.utils import load_graph_data, scaled_Laplacian, get_chebyshev_polynomials, load_traffic_data_windowed

class ModelManager:
    def __init__(self):
        self.model = None
        self.mean = 0.0
        self.std = 1.0
        self.lock = threading.Lock()
        self.status = "Initializing"

        adj_mx = load_graph_data(ADJ_PATH)
        L_tilde = scaled_Laplacian(adj_mx)
        cheb_poly = get_chebyshev_polynomials(L_tilde, K)
        self.cheb_poly_tensors = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_poly]

    def initialize_model(self):
        model = make_model(DEVICE, NB_BLOCK, IN_CHANNELS, K, NB_CHEV_FILTER, 
            NB_TIME_FILTER, TIME_STRIDES, self.cheb_poly_tensors, 
            NUM_FOR_PREDICT, LEN_INPUT, NUM_NODES)
        return model.to(DEVICE)

    def load_weights(self):
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            self.model = self.initialize_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.mean = checkpoint['mean']
            self.std = checkpoint['std']
            self.model.eval()
            self.status = "Ready"
            return True
        return False

    def train(self, window_size=None):
        with self.lock:
            self.status = "Training"

        print("Starting training process...")

        train_loader, test_loader, mean, std = load_traffic_data_windowed(DATA_PATH, BATCH_SIZE, LEN_INPUT, NUM_FOR_PREDICT, window_size)

        if self.model:
            print("Fine-tuning existing model...")
            training_net = deepcopy(self.model)
        else:
            print("Training from scratch...")
            training_net = self.initialize_model()

        optimizer = torch.optim.Adam(training_net.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        print("\n--- Počinjem treniranje ---")
        for epoch in range(EPOCHS):
            training_net.train()
            train_loss = 0

            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                output = training_net(x)
                output = output.transpose(1, 2)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            training_net.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    output = training_net(x)
                    output = output.transpose(1, 2)
                    loss = criterion(output, y)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(test_loader)

            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        with self.lock:
            self.model = training_net
            self.model.eval()
            self.mean = mean
            self.std = std
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'mean': self.mean,
                'std': self.std
            }, MODEL_PATH)
            self.status = "Ready"
            print("Training complete and model updated.")

    def predict(self, input_data):
        if self.model is None:
            return None

        with self.lock:
            try:
                norm_input = (input_data - self.mean) / self.std
                x_tensor = torch.FloatTensor(norm_input).to(DEVICE)
                x_tensor = x_tensor.transpose(0, 1)
                x_tensor = x_tensor.unsqueeze(0).unsqueeze(2)

                with torch.no_grad():
                    output = self.model(x_tensor)

                output = output.cpu().numpy()
                real_prediction = output * self.std + self.mean
                return real_prediction[0].tolist()
            except Exception as e:
                return None

    def append_data(self, new_measurements: list[list[float]]):
        for measurement in new_measurements:
            row_str = ",".join(map(str, measurement))
            with open(DATA_PATH, "a") as f:
                f.write(f"\n{row_str}")


model_manager = ModelManager()
