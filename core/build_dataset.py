from core.config import config
from core.datasets.cityscape import Cityscape 
import torch
from torch.utils.data import DataLoader
from core.datasets.cityscape import Cityscape
from core.datasets.nyu import NYU

def build_dataset(phase):
    data_file = config.dataset.train_data_file if phase == 'train' else config.dataset.val_data_file
    batch_size = config.train.batch_size if phase == 'train' else config.test.batch_size
    dataset = eval(config.dataset.name)(data_file, phase)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=phase=='train', num_workers=8)
    return data_loader
    #return dataset


