import torch

from torch.autograd import Variable

def torch_long(data, device):
    return Variable(torch.LongTensor(data)).to(device)

def torch_float(data, device):
    return Variable(torch.FloatTensor(data), requires_grad=True).to(device)
