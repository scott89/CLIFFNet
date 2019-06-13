from core.config import config
from core.datasets.cityscape import Cityscape 
import torch
import numpy as np
from torch.utils.data import DataLoader
from core.datasets.cityscape import Cityscape
from core.datasets.nyu import NYU

def _worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def build_dataset(phase):
    data_file = config.dataset.train_data_file if phase == 'train' else config.dataset.val_data_file
    batch_size = config.train.batch_size if phase == 'train' else config.test.batch_size
    dataset = eval(config.dataset.name)(data_file, phase)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=phase=='train', num_workers=8, worker_init_fn=_worker_init_fn)
    return data_loader
    #return dataset


