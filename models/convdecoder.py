import torch.nn as nn
import torch
from models import register


@register('convdecoder')
class ConvDecoder(nn.Module):

    def __init__(self, in_dim, out_dim,hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Conv2d(lastv, hidden, 1))
            layers.append(nn.PReLU())
            lastv = hidden
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Conv2d(lastv, out_dim, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)
        nn.init.normal_(self.out_layer.weight, std=0.01)

    def forward(self, x):
        x = self.layers(x)
        x = self.out_layer(x)
        return x