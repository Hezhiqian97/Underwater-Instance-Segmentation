
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PAttention(nn.Module):
    def __init__(self, dim):
        super(PAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)
        x1 = x1.unsqueeze(dim=2)
        x2 = torch.cat([x, x1], dim=2)
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        out = self.pa2(x2)
        out = self.sigmoid(out)
        return out



class CAPAF(nn.Module):
    def __init__(self,dim,reduction=8,m=-0.8):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 3,1,1 ,bias=True)
        self.sigmoid = nn.Sigmoid()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)

        self.w = w
        self.mix_block = nn.Sigmoid()


    def forward(self, x):
        l,m=x[0],x[1]
        initial=torch.cat([l,m],dim=1)
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        FC = sattn + cattn
        FC1 = self.sigmoid(self.pa(initial, FC))
        mix_factor = self.mix_block(self.w)
        out = initial * mix_factor.expand_as(initial) + FC1 * (1 - mix_factor.expand_as(FC1))
        x = self.conv(out)
        return x