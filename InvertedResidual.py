import torch
from torch import nn


def conv_dbl(in_dim, out_dim, stride):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU6(True)
    )


def conv_3_1(dim):
    return nn.Sequential(
        nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
        nn.BatchNorm2d(dim),
        nn.ReLU6(True),
        nn.Conv2d(dim, dim * 2, 1, 1, 0, bias=False),
        nn.BatchNorm2d(dim * 2),
        nn.ReLU6(True),
    )


def con1x1(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU6(True),
    )


def extend_layers(in_dim, inter_dim):
    return nn.Sequential(
        con1x1(in_dim, inter_dim),
        conv_3_1(inter_dim),
        con1x1(inter_dim * 2, inter_dim),
        conv_3_1(inter_dim),
        con1x1(inter_dim * 2, inter_dim),
    )


def output_layers(inter_dim, out_dim):
    return nn.Sequential(
        conv_3_1(inter_dim),
        nn.Conv2d(inter_dim * 2, out_dim, 1, 1, 0)
    )


class InvertedResidual(nn.Module):

    def __init__(self, in_dim, out_dim, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(expand_ratio * in_dim)
        self.use_res = stride == 1 and in_dim == out_dim

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 3, stride, 1, groups=in_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(True),

                nn.Conv2d(in_dim, out_dim, 1, 1, 0),
                nn.BatchNorm2d(out_dim)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(True),

                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(True),

                nn.Conv2d(hidden_dim, out_dim, 1, 1, 0),
                nn.BatchNorm2d(out_dim)
            )

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        else:
            return self.conv(x)
