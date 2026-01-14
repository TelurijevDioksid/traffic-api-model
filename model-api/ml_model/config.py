import torch
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATA_PATH = os.path.join(DATA_DIR, "vel.csv")              # csv putanja do podataka
ADJ_PATH = os.path.join(DATA_DIR, "adj.npz")               # putanja do matrice susjedstva
MODEL_PATH = os.path.join(DATA_DIR, "astgcn_model.pth")    # putanja za spremanje modela
SCALER_PATH = os.path.join(DATA_DIR, "scaler_params.json") # putanja za spremanje skalera

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

BATCH_SIZE = 64       
EPOCHS = 3
LEARNING_RATE = 0.001

NUM_NODES = 228       # Broj čvorova u grafu
LEN_INPUT = 12        # Duljina ulaznog vremenskog prozora, 12 * 5min = 1 sat
NUM_FOR_PREDICT = 12  # Duljina izlaznog vremenskog prozora za predvidanje, 12 * 5min = 1 sat
IN_CHANNELS = 1       # Broj ulaza po cvoru

NB_BLOCK = 1          # Broj ASTGCN blokova
K = 1                 # Red Chebyshev polinoma 
NB_CHEV_FILTER = 16   # Broj filtera u graf konvoluciji 
NB_TIME_FILTER = 16   # Broj filtera u vremenskoj konvoluciji 
TIME_STRIDES = 1      # Stride u vremenskoj konvoluciji

WINDOW_SIZE = 4537 # Veličina prozora za ponovno treniranje (zadnjih 24 sata)
END_OF_INITIAL_DATA_INDEX = 4537 # Indeks do kojeg su inicijalni podaci ucitani
