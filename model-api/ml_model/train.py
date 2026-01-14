import torch
import torch.nn as nn
import numpy as np
from model import make_model
from utils import load_traffic_data, load_graph_data, scaled_Laplacian, get_chebyshev_polynomials
from config import *

adj_mx = load_graph_data(ADJ_PATH)
L_tilde = scaled_Laplacian(adj_mx)
cheb_poly = get_chebyshev_polynomials(L_tilde, K)
cheb_poly_tensors = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_poly]

train_loader, test_loader, mean, std = load_traffic_data(DATA_PATH, BATCH_SIZE, LEN_INPUT, NUM_FOR_PREDICT)

net = make_model(DEVICE, NB_BLOCK, IN_CHANNELS, K, NB_CHEV_FILTER, 
                 NB_TIME_FILTER, TIME_STRIDES, cheb_poly_tensors, 
                 NUM_FOR_PREDICT, LEN_INPUT, NUM_NODES)
net.to(DEVICE)

optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

print("\n--- Počinjem treniranje ---")
for epoch in range(EPOCHS):
    net.train()
    train_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        output = net(x) 
        output = output.transpose(1, 2) 

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = net(x)
            output = output.transpose(1, 2)
            loss = criterion(output, y)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(test_loader)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

print("--- Treniranje završeno ---")

torch.save(net.state_dict(), "astgcn_model.pth")
print("Model spremljen kao 'astgcn_model.pth'")
