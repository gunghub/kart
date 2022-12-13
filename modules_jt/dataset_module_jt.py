
from torch.utils.data import Dataset, DataLoader
from modules_official import dense_transforms
import numpy as np
import random
from modules_jt.global_variables import *

class SuperTuxDatasetJT(Dataset):
    def __init__(self, dataset_name,track_name, transform=dense_transforms.ToTensor(),proportion=1.0,random_seed=100):
        
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        
        files=glob(path.join("dataset_base/"+dataset_name+"/"+track_name, '*.csv'))
        
        random.seed(random_seed)
        extracted_files=random.choices(files,k=int(len(files)*proportion))
        
        for f in extracted_files:
            i = Image.open(f.replace('.csv', '.png'))
            i.load()
            
            label=np.loadtxt(f, dtype=np.float32, delimiter=',')
            image, label=transform(i, label)
            
            self.data.append((image, label, track_name_to_code(track_name)))
            
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx]
    

def track_name_to_code(track_name):
    list = [
        'lighthouse', 
        'zengarden', 
        'hacienda', 
        'snowtuxpeak',
        'cornfield_crossing',
        'scotland',
        'cocoa_temple'
          ]
    
    return list.index(track_name)

def track_code_to_onehot(track_code):
    onehot=torch.zeros(NUM_TRACKS)
    onehot=onehot[track_code]=1
    return onehot