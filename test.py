from core.config import config
from core.datasets.cityscape import Cityscape
import torch
from torch.utils.data import DataLoader
from network.net import Net


data = Cityscape(config.dataset.train_data_file, 'train')
x = data[0]
data_loader = DataLoader(data, batch_size=4, num_workers=10)
iterator = data_loader.__iter__()

net = Net()
net.to(config.gpu[0])

a = data[0]
for i, b in enumerate(data_loader):
    if i == 10:
        break
    print b['gt'].shape
    b['gt'] = b['gt'].pin_memory().to(config.gpu[0], non_blocking=True)
    b['image'] = b['image'].pin_memory().to(config.gpu[0], non_blocking=True)
    pre = net(b['image'])
