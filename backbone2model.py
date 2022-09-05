import torch
from collections import OrderedDict

backbone_path = './assets/ghost.pt'
model_path = './assets/ghost.pt'
model = OrderedDict()
backbone = torch.load(backbone_path, map_location='cpu')['state_dict']
for k,v in backbone.items():
    model[k.replace('backbone.', '')] = backbone[k]

model.pop('head.weight')
torch.save(model, model_path)