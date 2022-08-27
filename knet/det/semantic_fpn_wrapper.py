import math


import torch
import torch.nn as nn
from torch.nn import init
from mmcv.cnn import ConvModule, normal_init
from mmdet.models.builder import NECKS, BACKBONES
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.utils import get_root_logger
from mmcv.ops import DeformConv2dPack
from mmcv.runner import BaseModule
import torch.nn.functional as F


@NECKS.register_module()
class SemanticFPNWrapper(nn.Module):
    """
    Implementation of Semantic FPN used in Panoptic FPN.

    Args:
        in_channels ([type]): [description]
        feat_channels ([type]): [description]
        out_channels ([type]): [description]
        start_level ([type]): [description]
        end_level ([type]): [description]
        cat_coors (bool, optional): [description]. Defaults to False.
        fuse_by_cat (bool, optional): [description]. Defaults to False.
        conv_cfg ([type], optional): [description]. Defaults to None.
        norm_cfg ([type], optional): [description]. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 start_level,
                 end_level,
                 cat_coors=False,
                 positional_encoding=None,
                 cat_coors_level=3,
                 fuse_by_cat=False,
                 return_list=False,
                 upsample_times=3,
                 with_pred=True,
                 num_aux_convs=0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 out_act_cfg=dict(type='ReLU'),
                 conv_cfg=None,
                 norm_cfg=None):
        super(SemanticFPNWrapper, self).__init__()

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.cat_coors = cat_coors
        self.cat_coors_level = cat_coors_level
        self.fuse_by_cat = fuse_by_cat
        self.return_list = return_list
        self.upsample_times = upsample_times
        self.with_pred = with_pred
        if positional_encoding is not None:
            self.positional_encoding = build_positional_encoding(
                positional_encoding)
        else:
            self.positional_encoding = None

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                if i == self.cat_coors_level and self.cat_coors:
                    chn = self.in_channels + 2
                else:
                    chn = self.in_channels
                if upsample_times == self.end_level - i:
                    one_conv = ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        inplace=False)
                    convs_per_level.add_module('conv' + str(i), one_conv)
                else:
                    for i in range(self.end_level - upsample_times):
                        one_conv = ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            padding=1,
                            stride=2,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                            inplace=False)
                        convs_per_level.add_module('conv' + str(i), one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    if i == self.cat_coors_level and self.cat_coors:
                        chn = self.in_channels + 2
                    else:
                        chn = self.in_channels
                    one_conv = ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        inplace=False)
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    if j < upsample_times - (self.end_level - i):
                        one_upsample = nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False)
                        convs_per_level.add_module('upsample' + str(j),
                                                   one_upsample)
                    continue

                one_conv = ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                convs_per_level.add_module('conv' + str(j), one_conv)
                if j < upsample_times - (self.end_level - i):
                    one_upsample = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module('upsample' + str(j),
                                               one_upsample)

            self.convs_all_levels.append(convs_per_level)

        if fuse_by_cat:
            in_channels = self.feat_channels * len(self.convs_all_levels)
        else:
            in_channels = self.feat_channels

        if self.with_pred:
            self.conv_pred = ConvModule(
                in_channels,
                self.out_channels,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                act_cfg=out_act_cfg,
                norm_cfg=self.norm_cfg)

        self.num_aux_convs = num_aux_convs
        self.aux_convs = nn.ModuleList()
        for i in range(num_aux_convs):
            self.aux_convs.append(
                ConvModule(
                    in_channels,
                    self.out_channels,
                    1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    act_cfg=out_act_cfg,
                    norm_cfg=self.norm_cfg))

    def init_weights(self):
        logger = get_root_logger()
        logger.info('Use normal intialization for semantic FPN')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def generate_coord(self, input_feat):
        x_range = torch.linspace(
            -1, 1, input_feat.shape[-1], device=input_feat.device)
        y_range = torch.linspace(
            -1, 1, input_feat.shape[-2], device=input_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([input_feat.shape[0], 1, -1, -1])
        x = x.expand([input_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        return coord_feat

    def forward(self, inputs):
        mlvl_feats = []
        for i in range(self.start_level, self.end_level + 1):
            input_p = inputs[i]
            if i == self.cat_coors_level:
                if self.positional_encoding is not None:
                    ignore_mask = input_p.new_zeros(
                        (input_p.shape[0], input_p.shape[-2],
                         input_p.shape[-1]),
                        dtype=torch.bool)
                    positional_encoding = self.positional_encoding(ignore_mask)
                    input_p = input_p + positional_encoding
                if self.cat_coors:
                    coord_feat = self.generate_coord(input_p)
                    input_p = torch.cat([input_p, coord_feat], 1)

            mlvl_feats.append(self.convs_all_levels[i](input_p))

        if self.fuse_by_cat:
            feature_add_all_level = torch.cat(mlvl_feats, dim=1)
        else:
            feature_add_all_level = sum(mlvl_feats)

        if self.with_pred:
            out = self.conv_pred(feature_add_all_level)
        else:
            out = feature_add_all_level

        if self.num_aux_convs > 0:
            outs = [out]
            for conv in self.aux_convs:
                outs.append(conv(feature_add_all_level))
            return outs

        if self.return_list:
            return [out]
        else:
            return out


@NECKS.register_module()
class UperNetAlignHead(BaseModule):

    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=256, feat_channels=256, align_types="v1",
                 start_level=1, end_level=3, conv3x3_type="conv", positional_encoding=None, cat_coors_level=3,
                 upsample_times=2, cat_coors=False, fuse_by_cat=False, return_list=False,
                 num_aux_convs=1, norm_cfg=dict(type='GN', num_groups=32, requires_grad=True) ):
        super(UperNetAlignHead, self).__init__()

        if positional_encoding is not None:
            self.positional_encoding = build_positional_encoding(
                positional_encoding)
        else:
            self.positional_encoding = None

        self.cat_coors_level = cat_coors_level
        self.align_types = align_types

        self.dcn = DeformConv2dPack(in_channels=256, out_channels=out_channels, kernel_size=3, padding=1)
        self.fpn_in = []
        for fpn_inplane in in_channels[:-1]:
            self.fpn_in.append(
                ConvModule(fpn_inplane, out_channels, kernel_size=1, norm_cfg=dict(type='BN2d'),
                           act_cfg=dict(type='ReLU'),
                           inplace=False)
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(in_channels) - 1):
            self.fpn_out.append(
                ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                           norm_cfg=dict(type='BN2d')))

            if conv3x3_type == 'conv':
                if self.align_types == "v1":
                    self.fpn_out_align.append(
                        AlignedModule(inplane=out_channels, outplane=out_channels // 2)
                    )
                else:
                    self.fpn_out_align.append(
                        AlignedModulev2PoolingAtten(inplane=out_channels, outplane=out_channels // 2)
                    )

            self.fpn_out = nn.ModuleList(self.fpn_out)
            self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

    def forward(self, conv_out):
        f = conv_out[-1]
        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))

        output_size = conv_out[1].size()[2:]
        fusion_list = []

        for i in range(0, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        x = fusion_list[0]
        for i in range(1, len(fusion_list)):
            x += fusion_list[i]

        # add position encodings
        ignore_mask = x.new_zeros(
                        (x.shape[0], x.shape[-2],
                         x.shape[-1]),
                        dtype=torch.bool)
        positional_encoding = self.positional_encoding(ignore_mask)
        x = x + positional_encoding

        return self.dcn(x)


class AlignedModule(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class AlignedModulev2PoolingAtten(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModulev2PoolingAtten, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :], flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)

        h_feature_mean = torch.mean(h_feature, dim=1).unsqueeze(1)
        l_feature_mean = torch.mean(low_feature, dim=1).unsqueeze(1)
        h_feature_max = torch.max(h_feature, dim=1)[0].unsqueeze(1)
        l_feature_max = torch.max(low_feature, dim=1)[0].unsqueeze(1)

        flow_gates = self.flow_gate(torch.cat([h_feature_mean, l_feature_mean, h_feature_max, l_feature_max], 1))

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


@BACKBONES.register_module()
class STDCNet1446(nn.Module):
    def __init__(self, base=64, layers=[4, 5, 3], block_num=4, type="cat", num_classes=1000, dropout=0.20,
                 pretrain_model='./pretrained_models/STDCNet1446_76.47.tar',
                 use_conv_last=False, norm_layer=nn.SyncBatchNorm, ):
        super(STDCNet1446, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block, norm_layer)
        self.conv_last = ConvX(base * 16, max(1024, base * 16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base * 16), max(1024, base * 16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base * 16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base * 16), num_classes, bias=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:6])
        self.x16 = nn.Sequential(self.features[6:11])
        self.x32 = nn.Sequential(self.features[11:])

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

        self.features = None
        self.conv_last = None
        self.gap = None
        self.fc = None
        self.bn = None
        self.relu = None
        self.dropout = None
        self.linear = None

    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model, map_location='cpu')["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block, norm_layer):
        features = []
        features += [ConvX(3, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2, norm_layer=norm_layer))
                elif j == 0:
                    features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2,
                                          norm_layer=norm_layer))
                else:
                    features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1,
                                          norm_layer=norm_layer))

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)

        return feat4, feat8, feat16, feat32


@BACKBONES.register_module()
class STDCNet813(nn.Module):
    def __init__(self, base=64, layers=[2, 2, 2], block_num=4, type="cat", num_classes=1000, dropout=0.20,
                 pretrain_model='./pretrained_models/STDCNet813_73.91.tar',
                 use_conv_last=False, norm_layer=nn.BatchNorm2d):
        super(STDCNet813, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block, norm_layer)
        self.conv_last = ConvX(base * 16, max(1024, base * 16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base * 16), max(1024, base * 16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base * 16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base * 16), num_classes, bias=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        self.x32 = nn.Sequential(self.features[6:])

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

        self.features = None
        self.conv_last = None
        self.gap = None
        self.fc = None
        self.bn = None
        self.relu = None
        self.dropout = None
        self.linear = None

    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model, map_location='cpu')["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block, norm_layer):
        features = []
        features += [ConvX(3, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2, norm_layer=norm_layer))
                elif j == 0:
                    features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2,
                                          norm_layer=norm_layer))
                else:
                    features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1,
                                          norm_layer=norm_layer))

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)

        return feat4, feat8, feat16, feat32




class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1, norm_layer=nn.BatchNorm2d):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                norm_layer(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                norm_layer(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                norm_layer(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1, norm_layer=nn.BatchNorm2d):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                norm_layer(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, norm_layer=nn.BatchNorm2d):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        self.bn = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out