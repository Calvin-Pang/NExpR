import torch.nn as nn
import torch
from models import register


@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, mode):
        super().__init__()
        self.mode = mode
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.PReLU())
            lastv = hidden
        self.layers = nn.Sequential(*layers)
        if self.mode == 'gray':
            self.out_layer = nn.Linear(lastv, out_dim)
        elif self.mode == 'rgb':
            self.out_layer_r = nn.Linear(lastv, out_dim)
            self.out_layer_g = nn.Linear(lastv, out_dim)
            self.out_layer_b = nn.Linear(lastv, out_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)
        if self.mode == 'gray':
            nn.init.normal_(self.out_layer.weight, std=0.01)
        elif self.mode == 'rgb':
            nn.init.normal_(self.out_layer_r.weight, std=0.01)
            nn.init.normal_(self.out_layer_g.weight, std=0.01)
            nn.init.normal_(self.out_layer_b.weight, std=0.01)

    def forward(self, x):
        shape = x.shape[:-1] # (bsize, G_h, G_w)
        x = self.layers(x.reshape(-1, x.shape[-1]))
        if self.mode == 'gray': 
            x = self.out_layer(x)
            return x.view(*shape, -1)
        elif self.mode == 'rgb':
            x_r = self.out_layer_r(x).unsqueeze(-2)
            x_g = self.out_layer_g(x).unsqueeze(-2)
            x_b = self.out_layer_b(x).unsqueeze(-2)
            x_out = torch.cat([x_r,x_g,x_b], dim = -2)
            return x_out.view(*shape, 3, x_out.shape[-1])