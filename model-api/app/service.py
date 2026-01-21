import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import threading
import random
from torch.utils.data import TensorDataset, DataLoader
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

        print("\n--- Start initial training ---")
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
                "model_state_dict": self.model.state_dict(),
                "mean": self.mean,
                "std": self.std
            }, MODEL_PATH)
            self.status = "Ready"
            print("--- Inital training complete ---")

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
        with self.lock:
            for measurement in new_measurements:
                row_str = ",".join(map(str, measurement))
                with open(DATA_PATH, "a") as f:
                    f.write(f"\n{row_str}")

    def _create_sequences(self, data, indices, len_input, num_predict):
        X_list = []
        Y_list = []

        for idx in indices:
            if idx - len_input < 0:
                continue
            if idx + num_predict > len(data):
                continue

            x_window = data[idx - len_input : idx, :]
            y_window = data[idx : idx + num_predict, :]
            X_list.append(x_window)
            Y_list.append(y_window)

        if not X_list:
            return None, None

        X = np.array(X_list)
        Y = np.array(Y_list)
        return torch.FloatTensor(X), torch.FloatTensor(Y)

    def online_train(self, data_per_step):
        if self.status != "Ready":
            print(f"Cannot start online training. Status is {self.status}")
            return

        with self.lock:
            self.status = "Training"

        try:
            print(f"--- Start online training ---")

            df = pd.read_csv(DATA_PATH, header=None) 
            full_data = df.values
            total_len = len(full_data)

            norm_data = (full_data - self.mean) / self.std

            start_new_idx = total_len - data_per_step
            new_indices = list(range(start_new_idx, total_len - NUM_FOR_PREDICT))

            if len(new_indices) == 0:
                print("Not enough data for online pass")
                self.status = "Ready"
                return

            # Za 48 novih i Batch Size 64, dodano je 3072 starih mjerenja
            num_history = (BATCH_SIZE - 1) * len(new_indices)
            history_pool = list(range(LEN_INPUT, start_new_idx))

            if len(history_pool) > 0:
                replace = len(history_pool) < num_history
                old_indices = np.random.choice(history_pool, size=num_history, replace=replace).tolist()
            else:
                old_indices = []

            all_indices = new_indices + old_indices
            random.shuffle(all_indices)
            X_batch, Y_batch = self._create_sequences(norm_data, all_indices, LEN_INPUT, NUM_FOR_PREDICT)

            if X_batch is None:
                print("Failed to create sequences.")
                self.status = "Ready"
                return

            X_batch = X_batch.transpose(1, 2).unsqueeze(2) # (Batch, Node, 1, Time)
            Y_batch = Y_batch.transpose(1, 2)

            dataset = TensorDataset(X_batch, Y_batch)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE) 
            criterion = nn.MSELoss()

            self.model.train()
            total_loss = 0
            for batch_idx, (x, y) in enumerate(loader):
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                optimizer.zero_grad()

                output = self.model(x)
                output = output.transpose(1, 2)

                loss = criterion(output, y)
                loss.backward()

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Done | Train loss: {avg_loss:.4f}")

            torch.save({
                "model_state_dict": self.model.state_dict(),
                "mean": self.mean,
                "std": self.std
            }, MODEL_PATH)

            print("--- Online training complete, saving checkpoint ---")

        except Exception as e:
            print(f"Error during online training: {e}")
            import traceback
            traceback.print_exc()

        finally:
            with self.lock:
                self.model.eval()
                self.status = "Ready"

model_manager = ModelManager()
