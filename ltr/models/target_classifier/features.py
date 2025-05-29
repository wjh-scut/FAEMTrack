import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
from ltr.models.layers.normalization import InstanceL2Norm
from ltr.models.layers.transform import InterpCat
import math
from torch.nn import init
from ltr.models.utils import conv_bn_relu, conv_bn, conv_gn_relu, conv_gn, CBAM


def residual_basic_block(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                         interp_cat=False, final_relu=False, init_pool=False):
    """Construct a network block based on the BasicBlock used in ResNet 18 and 34."""
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    if interp_cat:
        feat_layers.append(InterpCat())
    if init_pool:
        feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    for i in range(num_blocks):
        odim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        feat_layers.append(BasicBlock(feature_dim, odim))
    if final_conv:
        feat_layers.append(nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
        if final_relu:
            feat_layers.append(nn.ReLU(inplace=True))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))
    return nn.Sequential(*feat_layers)


def residual_basic_block_pool(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                              pool=True):
    """Construct a network block based on the BasicBlock used in ResNet."""
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    for i in range(num_blocks):
        odim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        feat_layers.append(BasicBlock(feature_dim, odim))
    if final_conv:
        feat_layers.append(nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
    if pool:
        feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))

    return nn.Sequential(*feat_layers)


def residual_bottleneck(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                        interp_cat=False, final_relu=False, final_pool=False, input_dim=None, final_stride=1):
    """Construct a network block based on the Bottleneck block used in ResNet."""
    if out_dim is None:
        out_dim = feature_dim
    if input_dim is None:
        input_dim = 4*feature_dim
    dim = input_dim
    feat_layers = []
    if interp_cat: #是否添加插值拼接层（InterpCat）
        feat_layers.append(InterpCat())
    for i in range(num_blocks):
        planes = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim // 4
        feat_layers.append(Bottleneck(dim, planes))
        dim = 4*feature_dim
    if final_conv:
        feat_layers.append(nn.Conv2d(dim, out_dim, kernel_size=3, padding=1, bias=False, stride=final_stride))
        if final_relu:
            feat_layers.append(nn.ReLU(inplace=True))
        if final_pool:
            feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))
    return nn.Sequential(*feat_layers)

class FAEMTrack(nn.Module):
    def __init__(self, in_dim=1024, out_dim=512, norm_scale=1.0):
        super(FAEMTrack, self).__init__()

        ratio = 2
        inter_dim = in_dim // ratio
        oup = in_dim * ratio
        # dw_sizes = [3, 5, 7]
        # new_channels_per_branch = inter_dim * (ratio - 1)

        dw_sizes = [3, 5, 7, 9, 11]
        out_chs = [132, 219, 307, 394, 484]

        self.gate_fn = nn.Sigmoid()
        self.primary_conv = conv_bn_relu(in_dim, inter_dim, kernel_size=1, padding=0, bias=True)

        self.cheap_ops = nn.ModuleList()
        for dw, oc in zip(dw_sizes, out_chs):
            branch = nn.Sequential(
                # 1) depthwise: in_channels->in_channels, groups=in_channels
                nn.Conv2d(
                    in_channels=inter_dim,
                    out_channels=inter_dim,  # 必须等于 in_channels
                    kernel_size=dw,
                    stride=1,
                    padding=dw // 2,
                    groups=inter_dim,  # depthwise
                    bias=False
                ),
                nn.BatchNorm2d(inter_dim),
                nn.ReLU(inplace=True),

                # 2) pointwise: inter_dim->oc
                nn.Conv2d(
                    in_channels=inter_dim,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            )
        # for dw in dw_sizes:
        #     branch = nn.Sequential(
        #         # 使用深度卷积：groups=init_channels 确保每个通道独立运算
        #         nn.Conv2d(inter_dim, new_channels_per_branch, dw, 1, dw // 2, groups=inter_dim, bias=False),
        #         nn.BatchNorm2d(new_channels_per_branch),
        #         nn.ReLU(inplace=True),
        #     )
            self.cheap_ops.append(branch)

        self.short_conv = nn.Sequential(
            nn.Conv2d(in_dim, oup, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
            nn.BatchNorm2d(oup),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_dim * ratio, out_dim, kernel_size=3, padding=1, bias=False),
            InstanceL2Norm(scale=norm_scale))

    def forward(self, x):
        res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
        x1 = self.primary_conv(x)
        x2 = [branch(x1) for branch in self.cheap_ops]
        #x2 = self.cheap_operation(x1)
        fused_out = torch.cat([x1]+x2, dim=1)
        fused_out = fused_out * F.interpolate(self.gate_fn(res), size=(fused_out.shape[-2], fused_out.shape[-1]), mode='nearest')
        fused_out = self.final_conv(fused_out)
        return fused_out
######################################################################################################################MultiScaleGhostv2_1


if __name__ == '__main__':
    t = residual_bottleneck(num_blocks=0,final_conv=True,out_dim=512)
    norm_scale = math.sqrt(1.0 / (256))
    m = FAEMTrack()
    input = torch.randn(32,1024,58,58)
    output = m(input)
    print(output.shape)



