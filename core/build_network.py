import torch
from torch.nn import DataParallel
from core.config import config
from network.net import Net
from network.vgg import vgg16_d3




def build_network():
    net = Net()
    net = DataParallel(net, device_ids=config.gpu).to(config.gpu[0])
    return net


def build_loss_network(snapshot):
    net = vgg16_d3(inter_output_layer=[0, 5])
    net = DataParallel(net, device_ids=config.gpu).to(config.gpu[0])
    # load parameters
    ckpt = torch.load(snapshot)
    net.load_state_dict(ckpt['model_state_dict'], strict=False)
    # freeze loss network parameters
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


