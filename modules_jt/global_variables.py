PROJECT_ROOT="/home/phuang/kart/dev2"
FILE_PATH=PROJECT_ROOT+'/abc.py'
import torch
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_TRACKS=7