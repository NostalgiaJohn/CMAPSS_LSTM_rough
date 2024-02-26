import torch

N_HIDDEN = 96  # NUMBER OF HIDDEN STATES
N_LAYER = 4  # NUMBER OF LSTM LAYERS
N_EPOCH = 100  # NUM OF EPOCHS
MAX = 135  # UPPER BOUND OF RUL
LR = 0.01  # LEARNING RATE

# GPU support
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# CPU
# DEVICE = "cpu"