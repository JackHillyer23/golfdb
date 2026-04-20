from model import EventDetector

m = EventDetector(pretrain=False, width_mult=1., lstm_layers=1, lstm_hidden=256, bidirectional=True, dropout=False)

print("=== Top level children ===")
for i, (name, child) in enumerate(m.named_children()):
    print(i, name, type(child))

print("\n=== CNN children ===")
for i, (name, child) in enumerate(m.cnn.named_children()):
    print(i, name, type(child))

print("\n=== Checking what k=10 actually freezes ===")
from util import freeze_layers
freeze_layers(10, m)
for name, param in m.named_parameters():
    print(name, 'requires_grad:', param.requires_grad)

m2 = EventDetector(pretrain=True, width_mult=1., lstm_layers=1, lstm_hidden=256, bidirectional=True, dropout=False)
# check first weight to see if pretrained weights loaded
import torch
print(m2.cnn[0][0].weight[0,0])  # should not be all zeros if pretrained loaded