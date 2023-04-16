import torch
import sys
from torchvision import models
from torchinfo import summary
from pprint import pprint

types = set()

def rep(d):
    if isinstance(d, dict):
        for key in d:
            types.add(type(d[key]))
            if isinstance(d[key], torch.Tensor):
                d[key] = d[key].shape
            else:
                if isinstance(d[key], tuple):
                    d[key] = list(d[key])
                rep(d[key])
    elif isinstance(d, list):
        for i in range(len(d)):
            types.add(type(d[i]))
            if isinstance(d[i], torch.Tensor):
                d[i] = d[i].shape
            else:
                rep(d[i])
    types.add(type(d))

model_name = sys.argv[1]
model = torch.load(model_name)
rep(model)

for key in model["state_dict"]:
    print(key.ljust(80), model["state_dict"][key])
