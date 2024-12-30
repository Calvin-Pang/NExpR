import torch.nn as nn
from .mlp import *
from .convdecoder import *
from .edsr import *
from utils import *
from einops import rearrange
import time

@register('nexpr')
class NExpR(nn.Module):

    def __init__(self, bar_size, degree, max_basis, mode, learnable_basis = True, neg_freq = True, cell_decode=True, pos_encode=True):
        super().__init__()
        self.bar_size = bar_size
        self.degree = degree
        self.length_basis = degree+1
        self.max_basis = max_basis
        self.mode = mode
        self.cell_decode = cell_decode
        self.pos_encode = pos_encode
        self.channel = 1 if mode == 'gray' else 3
        self.encoder_feature_num = 64
        # self.tran_feature_num = 64
        # if pos_encode: self.tran_feature_num += 2
        self.decoder_feature_num = self.encoder_feature_num * (self.bar_size) ** 2
        if cell_decode: self.decoder_feature_num += 2
        self.neg_freq = neg_freq
        self.learnable_basis = learnable_basis
        if self.learnable_basis:
            self.basis_x = nn.Parameter((max_basis/degree) * torch.ones(degree))
            self.basis_y = nn.Parameter((max_basis/degree) * torch.ones(degree))
        else:
            self.basis_x = (max_basis/degree) * torch.ones(degree)
            self.basis_y = (max_basis/degree) * torch.ones(degree)   

        self.encoder = make_edsr_baseline(rgb_range = 1, n_feats = self.encoder_feature_num, no_upsampling = True, n_colors = 1)
        self.decoder = ConvDecoder(in_dim = self.decoder_feature_num, out_dim = 2 * (self.length_basis) **2, hidden_list=[256,256,256,256])
        self.function_map = None
        self.tran_layer = nn.Conv2d(in_channels=self.encoder_feature_num, out_channels=self.encoder_feature_num*(self.bar_size)**2,
                                    kernel_size=self.bar_size, stride=self.bar_size)


    def cal_x_y_coord(self):
        if self.neg_freq:
            self.x_coord = torch.cat([torch.cumsum(self.basis_x[:int(self.degree/2)], 0)-self.max_basis/2, 
                                    torch.zeros(1).to(self.basis_x), torch.cumsum(self.basis_x[int(self.degree/2):], 0)]) * torch.pi
            self.y_coord = torch.cat([torch.cumsum(self.basis_y[:int(self.degree/2)], 0)-self.max_basis/2, 
                                    torch.zeros(1).to(self.basis_x), torch.cumsum(self.basis_y[int(self.degree/2):], 0)]) * torch.pi
        else:
            self.x_coord = torch.cat([torch.cumsum(self.basis_x, 0), torch.zeros(1).to(self.basis_x)]) * torch.pi
            self.y_coord = torch.cat([torch.cumsum(self.basis_y, 0), torch.zeros(1).to(self.basis_x)]) * torch.pi
        
    def gen_feat(self, inp):
        # inp: bsize,1,h,w
        inp_pad, _ = pad_tensor(inp, self.bar_size)
        feat_maps = self.encoder(inp_pad) 
        self.feat = feat_maps#F.unfold(feat_maps, 3, padding=1).view(feat_maps.shape[0], feat_maps.shape[1] * 9, feat_maps.shape[2], feat_maps.shape[3])
        return self.feat

    def get_function_map(self, cell = None):
        # self.feat: bsize, C, H, W
        feat = self.feat #.permute(0, 2, 3, 1) # bsize, h, w, c
        feat_patch = self.tran_layer(feat) # bsize, H, W, C patch level
        # feat_max = self.max_layer(feat)
        # feat_avg = self.avg_layer(feat)
        # feat_patch = self.tran_layer(torch.cat([feat_max, feat_avg], dim = 1))
        (bsize, C, H, W) = feat_patch.shape
        if self.pos_encode:
            pe = positionalencoding2d(C,H,W).to(feat_patch)
            feat_patch += pe
        if cell is not None:
            cell = cell.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
            feat_patch = torch.cat([feat_patch, cell], dim=1)
        self.function_map = self.decoder(feat_patch).permute(0, 2, 3, 1)
        return self.function_map   # gray: bsize, H_fm(G_h), W_fm(G_w), degree**2*2 
                                                     # rgb:  bsize, H_fm(G_h), W_fm(G_w), 3, degree 

    def make_coords(self, x_lst, y_lst):
        # x_lst: bsize, sample, bar_size, 1
        batch_size, num_sample = x_lst.shape[0], x_lst.shape[1]
        x_basis = (self.x_coord).to(x_lst) * x_lst.repeat(1,1,1,self.length_basis)
        y_basis = (self.y_coord).to(x_lst) * y_lst.repeat(1,1,1,self.length_basis)
        x_basis = x_basis.view(batch_size, num_sample,1,self.length_basis,1).expand(-1,-1,-1,-1,self.length_basis)
        y_basis = y_basis.view(batch_size, num_sample,1,1,self.length_basis).expand(-1,-1,-1,self.length_basis,-1)
        basis_2d = (x_basis + y_basis).view(batch_size, num_sample, 1, -1)
        x_out = torch.concat([torch.sin(basis_2d), torch.cos(basis_2d)], dim = -1)
        return x_out

    def query_intensity(self, coord):
        batch_size = self.function_map.shape[0]
        num_sample = coord.shape[1]
        
        global_x = coord[:, :, 0]  # size [B, 2304]
        global_y = coord[:, :, 1]  # size [B, 2304]
        h_coord_index = (global_x // self.bar_size).long().unsqueeze(-1) # 32,2304,1
        w_coord_index = (global_y // self.bar_size).long().unsqueeze(-1) # 32,2304,1
        batch_index = torch.arange(batch_size).view(-1, 1, 1)
        batch_index = batch_index.expand(-1, num_sample, 1)
        params = self.function_map[batch_index, h_coord_index, w_coord_index, :] # 32, 2304, ?, degree
        local_x = (global_x  - h_coord_index.squeeze(-1) * self.bar_size).unsqueeze(-1)  # 32, 2304, 1
        local_y = (global_y  - w_coord_index.squeeze(-1) * self.bar_size).unsqueeze(-1)  # 32, 2304, 1
        query_coord = self.make_coords(local_x.unsqueeze(-1), local_y.unsqueeze(-1)) # 32, 2304, 1, 21
        if self.mode == 'gray': 
            # time_s = time.time()
            # for i in range(1000):
            result = torch.sum(query_coord * params, dim = -1)
            # time_e = time.time()
        elif self.mode == 'rgb':
            result = torch.einsum('abcd,abcd->abc', query_coord.repeat(1,1,3,1), params.squeeze(-3)) 
        return result #, time_e - time_s

    def forward(self, inp, coord, cell = None):
        if self.training: self.cal_x_y_coord()
        self.gen_feat(inp)
        self.get_function_map(cell)
        
        res = self.query_intensity(coord)
        
        return res#, time_used
