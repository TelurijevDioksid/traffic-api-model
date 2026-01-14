import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from model import make_model
from utils import load_traffic_data, load_graph_data, scaled_Laplacian, get_chebyshev_polynomials
from config import *

MODEL_PATH = "astgcn_model.pth"

def main():
    adj_mx = load_graph_data(ADJ_PATH)
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_poly = get_chebyshev_polynomials(L_tilde, K)
    cheb_poly_tensors = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_poly]

    _, test_loader, mean, std = load_traffic_data(DATA_PATH, BATCH_SIZE, LEN_INPUT, NUM_FOR_PREDICT)

    net = make_model(DEVICE, NB_BLOCK, IN_CHANNELS, K, NB_CHEV_FILTER, 
                     NB_TIME_FILTER, TIME_STRIDES, cheb_poly_tensors, 
                     NUM_FOR_PREDICT, LEN_INPUT, NUM_NODES)
    net.to(DEVICE)

    print(f"Učitavam model iz: {MODEL_PATH}")
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.eval() 

    all_preds = []
    all_targets = []

    print("Generiram predviđanja...")
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            output = net(x)
            output = output.transpose(1, 2) 

            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    print(f"Shape predviđanja: {all_preds.shape}")

    real_preds = all_preds * std + mean
    real_targets = all_targets * std + mean

    mae = mean_absolute_error(real_targets.flatten(), real_preds.flatten())
    rmse = np.sqrt(mean_squared_error(real_targets.flatten(), real_preds.flatten()))

    print("\n--- REZULTATI TESTIRANJA ---")
    print(f"MAE (Srednja apsolutna greška): {mae:.2f}")
    print(f"RMSE (Korijen srednje kvadratne greške): {rmse:.2f}")
    print("------------------------------")


if __name__ == "__main__":
    main()