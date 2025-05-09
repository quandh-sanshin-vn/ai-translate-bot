import torch
print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is {device}')

#fix cuda error
#find cuda version
#nvidia-smi
#NVIDIA-SMI 552.55                 Driver Version: 552.55         CUDA Version: 12.4  

#pip uninstall torch
#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124