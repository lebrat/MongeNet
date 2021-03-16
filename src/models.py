import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LowDistotionParametrization(nn.Module):
    # Low distortion method from Heitz, Eric. "A Low-Distortion Map Between Triangle and Square." (2019).

    def __init__(self, cfg):
        super().__init__()

    def forward(self, r, x):        
        b1, t1, c1 = x.shape
        b2, n, c2 = r.shape
        assert t1 == 3 and c1 == 2 and b1 == b2 and c2 == 2, "Inconsistent input shapes in the parametrization {}, {}".format(x.shape, r.shape) 
        
        r1, r2 = r[:, :, 0].view(b2, n, 1), r[:, :, 1].view(b2, n, 1)
        v1 , v2, v3 = x[:, 0, :].view(b1, 1, c1), x[:, 1, :].view(b1, 1, c1), x[:, 2, :].view(b1, 1, c1)

        mask = r2 > r1
        r1 = mask * (r1 * 0.5) + (~mask) * (r1 - 0.5*r2)
        r2 = mask * (r2 - r1) +  (~mask) * (r2 * 0.5)
        out = r1 * v1 + r2 * v2 + (1.-r1-r2) * v3
        return out


class  ParallelogramParametrization(nn.Module):
    # Parallelogram parametrization from https://mathworld.wolfram.com/TrianglePointPicking.html    
    
    def __init__(self, cfg):
        super().__init__()

    def forward(self, r, x):
        b1, t1, c1 = x.shape
        b2, n, c2 = r.shape
        assert t1 == 3 and c1 == 2 and b1 == b2 and c2 == 2, "Inconsistent input shapes in the parametrization {}, {}".format(x.shape, r.shape) 
        sqr_r1, r2 = torch.sqrt(r[:, :, 0]).view(b2, n, 1), r[:, :, 1].view(b2, n, 1)
        v1 , v2, v3 = x[:, 0, :].view(b1, 1, c1), x[:, 1, :].view(b1, 1, c1), x[:, 2, :].view(b1, 1, c1)
        out = (1. - sqr_r1) * v1 + (1. - r2) * sqr_r1 * v2 + sqr_r1 * r2 * v3
        return out


class MongeNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # model hyper-parameters
        self.max_num_points = int(cfg.mongenet.max_points)
        self.param_name = str(cfg.mongenet.parametrization)
        self.dropout_ratio = float(cfg.mongenet.dropout)
        self.random_input = str(cfg.mongenet.random_input)
        self.arch_type = str(cfg.mongenet.arch_type)
        self.device = str(cfg.mongenet.device)

        # pack all point-clouds prediction
        self.num_outputs = int(((1 + self.max_num_points)*self.max_num_points)/2) 
        self.output_index = torch.Tensor(sum([[i] * i for i in range(1, self.max_num_points+1)], [])).long().to(self.device)
        
        # set network arch to use
        self.layers = {
            '2_hl': nn.Sequential(
                nn.Linear(3,128), nn.ReLU(True), nn.Dropout(self.dropout_ratio), 
                nn.Linear(128, self.num_outputs*2), nn.Sigmoid()),
            '3_hl': nn.Sequential(
                nn.Linear(3, 64), nn.ReLU(True), nn.Dropout(self.dropout_ratio),
                nn.Linear(64, 64), nn.ReLU(True), nn.Dropout(self.dropout_ratio),
                nn.Linear(64, self.num_outputs*2), nn.Sigmoid()),
            '4_hl': nn.Sequential(
                nn.Linear(3, 64), nn.ReLU(True), nn.Dropout(self.dropout_ratio),
                nn.Linear(64, 64), nn.ReLU(True), nn.Dropout(self.dropout_ratio),
                nn.Linear(64, 64), nn.ReLU(True), nn.Dropout(self.dropout_ratio),
                nn.Linear(64, self.num_outputs*2), nn.Sigmoid()),
            }[self.arch_type]
        
        # set unit-square to triangle parametrization
        self.parametrization = {
            'parallelogram': ParallelogramParametrization(cfg),
            'lowdistortion': LowDistotionParametrization(cfg),
        }[self.param_name]
        
        # set random-input format
        self.random_input_func = {
            'normal': lambda bsize: torch.clamp(torch.randn(bsize, 1), -1.0, 1.0).float().to(self.device),
            'uniform': lambda bsize: torch.clamp(torch.rand(bsize, 1)*2. - 1., -1.0, 1.0).float().to(self.device),
        }[self.random_input]

    def forward(self, x):
        b, t, c = x.shape
        assert t == 3 and c == 2, "Wrong input shape to the network {}".format(x.shape) 
        out = torch.cat([x[:, -1], self.random_input_func(b)], dim=-1)
        out = self.layers(out)
        out = out.view(b, self.num_outputs, 2)
        out = self.parametrization(out, x)              
        return out, self.output_index