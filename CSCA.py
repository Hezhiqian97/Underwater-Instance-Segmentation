
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Mix(nn.Module):

    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()
    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)

        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class channlAttention(nn.Module):

    def __init__(self, channel, b=1, gamma=2):
        super(channlAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, input):
        U = self.avg_pool(input)
        Ugc = self.conv1(U.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)  # (batch_size, channels, 1)
        Ulc = self.fc(U).squeeze(-1).transpose(-1, -2)  # (batch_size, 1, channels)
        out1 = torch.sum(torch.matmul(Ugc, Ulc), dim=1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, channels, 1, 1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(Ulc.transpose(-1, -2), Ugc.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        out = self.mix(out1, out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        W = self.sigmoid(out)

        return input * W


class CSCA(nn.Module):
    def __init__(self, embed_dim, squeezes=(4, 4), shuffle=4, expan_att_chans=4,group_kernel_sizes: t.List[int] = [3, 5, 7, 9],attn_bias=False,proj_drop=0.):
        super(CSCA, self).__init__()
        self.embed_dim = embed_dim
        self.group_chans = group_chans = embed_dim // 4
        self.att1 = nn.Softmax(dim=2)
        self.att2= nn.Softmax(dim=3)

        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, )
        self.g_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, )
        self.g_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, )
        self.g_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, )
        self.norm_h = nn.GroupNorm(4, embed_dim)
        self.norm_w = nn.GroupNorm(4, embed_dim)

        self.chan=channlAttention(embed_dim)

        self.norm = LayerNorm(embed_dim, eps=1e-6)
        self.dwc = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, )
        self.ct = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, )
        self.proj = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, )
        self.a=nn.Sigmoid()
        self.q = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, )
        self.k = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, )

    def forward(self, x):
        x = self.norm(x)

        b, c, h_, w_ = x.size()

        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # (B, C, W)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.g_s(g_x_h_s),
            self.g_m(g_x_h_m),
            self.g_l(g_x_h_l),
        ), dim=1))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.g_s(g_x_w_s),
            self.g_m(g_x_w_m),
            self.g_l(g_x_w_l)
        ), dim=1))
        x_w_attn = x_w_attn.view(b, c, 1, w_)
        Q1 = self.ct((x*self.a(x_h_attn + x_w_attn)))
        k1 = self.chan(x)
        Q=self.q(Q1)
        K= self.k(k1)
        out = self.proj(self.a(Q * K) * x)
        return out



