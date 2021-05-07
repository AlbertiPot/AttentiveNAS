# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation adapted from OFA: https://www.google.com/search?q=once+for+all+github

import copy
import random
import collections
import math

import torch
import torch.nn as nn

from .modules.dynamic_layers import DynamicMBConvLayer, DynamicConvBnActLayer, DynamicLinearLayer, DynamicShortcutLayer
from .modules.static_layers import MobileInvertedResidualBlock
from .modules.nn_utils import make_divisible, int2list
from .modules.nn_base import MyNetwork
from .attentive_nas_static_model import AttentiveNasStaticModel

# target：根据解空间的参数生成最大体积的超网，
# opts: 超网的深度runtime_depth在这里设定，将block的选择list传给dynamic layers去生成最大的block
# opts: sample_active_subnet_within_range采样特定的网络（从cfg_candidates中采样）
# opts: 由set_active_subnet设置采样的子网的参数（包括采样后更新的block参数,即修改dynamic layers中的active_out_channel active_kernel_size active_expand_ratio），由forward运行
# param：cfg_candidates存放解空间，runtime_depth控制网络的深度，其他如dropout ratio等
class AttentiveNasDynamicModel(MyNetwork):

    def __init__(self, supernet, n_classes=1000, bn_param=(0., 1e-5)):
        super(AttentiveNasDynamicModel, self).__init__()

        self.supernet = supernet                                    # 解空间，对应yaml.supernet_config部分
        self.n_classes = n_classes
        self.use_v3_head = getattr(self.supernet, 'use_v3_head', False)
        self.stage_names = ['first_conv', 'mb1', 'mb2', 'mb3', 'mb4', 'mb5', 'mb6', 'mb7', 'last_conv']

        self.width_list, self.depth_list, self.ks_list, self.expand_ratio_list = [], [], [], []
        for name in self.stage_names:                               # 遍历stage：first_conv, mb1, ... , last_conv
            block_cfg = getattr(self.supernet, name)                # 取出stage对应的字典，如mb1对应的{c:[],d:[],k:[]...}
            
            self.width_list.append(block_cfg.c)# => list[list]      # 提取通道数c,层数d，核大小k，扩张率t，后三个仅提取mb的，因为对于first和last conv，可供选择的仅有通道数
            if name.startswith('mb'):
                self.depth_list.append(block_cfg.d)
                self.ks_list.append(block_cfg.k)
                self.expand_ratio_list.append(block_cfg.t)
        self.resolution_list = self.supernet.resolutions

        # 搜索空间作为采样空间
        self.cfg_candidates = {
            'resolution': self.resolution_list ,
            'width': self.width_list,
            'depth': self.depth_list,
            'kernel_size': self.ks_list,
            'expand_ratio': self.expand_ratio_list
        }

        #first conv layer, including conv, bn, act:  first layer → 使用DynamicConvbnActLayer
        out_channel_list, act_func, stride = \
            self.supernet.first_conv.c, self.supernet.first_conv.act_func, self.supernet.first_conv.s
        self.first_conv = DynamicConvBnActLayer( 
            in_channel_list=int2list(3), out_channel_list=out_channel_list, # input_channel为 [3] 
            kernel_size=3, stride=stride, act_func=act_func,
        )

        # inverted residual blocks
        self.block_group_info = []
        blocks = []                                                 # module list，暂存，后期放入ModuleList
        _block_index = 0
        feature_dim = out_channel_list                              # 接first layer的输出通道数，=> list
        for stage_id, key in enumerate(self.stage_names[1:-1]):     # 去掉first和final conv layer
            block_cfg = getattr(self.supernet, key)                 # 遍历mb1到mb7
            width = block_cfg.c
            n_block = max(block_cfg.d)                              # 层数layers [1,2] 选择最大的做指引
            act_func = block_cfg.act_func
            ks = block_cfg.k
            expand_ratio_list = block_cfg.t
            use_se = block_cfg.se

            self.block_group_info.append([_block_index + i for i in range(n_block)]) # block_group_info存储了每个stage最大block数目的按顺序index -> [[0,1],[2,3,4],[5,6,7,8]....]
            _block_index += n_block

            output_channel = width # => list
            for i in range(n_block):                                # 遍历一个stage中最大blocks数
                stride = block_cfg.s if i == 0 else 1               # 第一个block的stride赋值解空间的指定值，其他block stride =1

                if min(expand_ratio_list) >= 4:
                    expand_ratio_list = [_s for _s in expand_ratio_list if _s >= 4] if i == 0 else expand_ratio_list # 当扩张率list中最小值≥4，对于第一block所有的扩张率应≥4
                mobile_inverted_conv = DynamicMBConvLayer(          # mobile_inverted_conv → DynamicMBConvLayer
                    in_channel_list=feature_dim, # => list
                    out_channel_list=output_channel, 
                    kernel_size_list=ks,
                    expand_ratio_list=expand_ratio_list, 
                    stride=stride, 
                    act_func=act_func, 
                    use_se=use_se,
                    channels_per_group=getattr(self.supernet, 'channels_per_group', 1)
                )
                shortcut = DynamicShortcutLayer(feature_dim, output_channel, reduction=stride)  # shortcut → DynamicShortcutLayer
                blocks.append(MobileInvertedResidualBlock(mobile_inverted_conv, shortcut))      # 用MobileInvertedResidualBlock将 mobile_inverted_conv 和 shortcut合并成为MBlock
                feature_dim = output_channel
        self.blocks = nn.ModuleList(blocks)

        # final conv layer → DynamicConvBnActLayer
        last_channel, act_func = self.supernet.last_conv.c, self.supernet.last_conv.act_func
        if not self.use_v3_head:
            self.last_conv = DynamicConvBnActLayer(
                    in_channel_list=feature_dim, out_channel_list=last_channel,
                    kernel_size=1, act_func=act_func,
            )
        else:
            expand_feature_dim = [f_dim * 6 for f_dim in feature_dim]
            self.last_conv = nn.Sequential(collections.OrderedDict([
                ('final_expand_layer', DynamicConvBnActLayer(
                    feature_dim, expand_feature_dim, kernel_size=1, use_bn=True, act_func=act_func)
                ),
                ('pool', nn.AdaptiveAvgPool2d((1,1))),
                ('feature_mix_layer', DynamicConvBnActLayer(
                    in_channel_list=expand_feature_dim, out_channel_list=last_channel,
                    kernel_size=1, act_func=act_func, use_bn=False,)
                ),
            ]))

        #classifier layer → DynamicLinearLayer
        self.classifier = DynamicLinearLayer(
            in_features_list=last_channel, out_features=n_classes, bias=True
        )

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1]) # default: bn_param = (0., 1e-5), bn除了仿射的两个参数，还有两个参数：momentum用于更新x，eps用于防止分母为0

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info] # -> list, 存储各个stage的最大blocks数 [2,3,4, ...]

        self.zero_residual_block_bn_weights()

        self.active_dropout_rate = 0
        self.active_drop_connect_rate = 0
        self.active_resolution = 224


    def zero_residual_block_bn_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, MobileInvertedResidualBlock):
                    if isinstance(m.mobile_inverted_conv, DynamicMBConvLayer) and m.shortcut is not None:
                        m.mobile_inverted_conv.point_linear.bn.bn.weight.zero_() # 清零beta


    @staticmethod
    def name():
        return 'AttentiveNasModel'


    def forward(self, x):
        # resize input to target resolution first 判断输入最后一维是不是默认是224，否则插值到224
        if x.size(-1) != self.active_resolution:
            x = torch.nn.functional.interpolate(x, size=self.active_resolution, mode='bicubic')

        # first conv
        x = self.first_conv(x)
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id] # 深度
            active_idx = block_idx[:depth] # 取出特定深度的层id
            for idx in active_idx:
                x = self.blocks[idx](x)

        x = self.last_conv(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling 先对dim=3的W做平均，再对dim=2的H做平均
        x = torch.squeeze(x) # 去掉某个维度上=1的维度

        if self.active_dropout_rate > 0 and self.training:
            x = torch.nn.functional.dropout(x, p = self.active_dropout_rate)

        x = self.classifier(x)
        return x


    @property                   # 以访问属性的方法实现getter(), eg. 实例名+module_str 而不用实例名.方法名
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str += self.blocks[0].module_str + '\n'

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'
        if not self.use_v3_head:
            _str += self.last_conv.module_str + '\n'
        else:
            _str += self.last_conv.final_expand_layer.module_str + '\n'
            _str += self.last_conv.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': AttentiveNasDynamicModel.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'last_conv': self.last_conv.config if not self.use_v3_head else None,
            'final_expand_layer': self.last_conv.final_expand_layer if self.use_v3_head else None,
            'feature_mix_layer': self.last_conv.feature_mix_layer if self.use_v3_head else None,
            'classifier': self.classifier.config,
            'resolution': self.active_resolution
        }


    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')


    """ set, sample and get active sub-networks """
    # target: 根据采样需要，重设上面的网络参数
    def set_active_subnet(self, resolution=224, width=None, depth=None, kernel_size=None, expand_ratio=None, **kwargs):
        assert len(depth) == len(kernel_size) == len(expand_ratio) == len(width) - 2                            # -2 去掉first和final layer的长度
        #set resolution
        self.active_resolution = resolution

        # first conv
        self.first_conv.active_out_channel = width[0] 

        for stage_id, (c, k, e, d) in enumerate(zip(width[1:-1], kernel_size, expand_ratio, depth)):
            start_idx, end_idx = min(self.block_group_info[stage_id]), max(self.block_group_info[stage_id])     # 每个stage中开始和结束的block的index
            for block_id in range(start_idx, start_idx + d): # 遍历每个stage深度为d的block
                block = self.blocks[block_id]
                #block output channels 输出的通道数固定为c
                block.mobile_inverted_conv.active_out_channel = c
                if block.shortcut is not None:
                    block.shortcut.active_out_channel = c

                #dw kernel size
                block.mobile_inverted_conv.active_kernel_size = k

                #dw expansion ration
                block.mobile_inverted_conv.active_expand_ratio = e

        # target: 比较各个stage的depth与超网的最大深度，找最小值来更改运行时的深度做采样
        # note: 在上面初始stage中的block中后再在这里设置每个stage的深度，可能会有些预设置的block没有用到
        #IRBlocks repated times
        for i, d in enumerate(depth):
            self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

        #last conv
        if not self.use_v3_head:
            self.last_conv.active_out_channel = width[-1]
        else:
            # default expansion ratio: 6
            self.last_conv.final_expand_layer.active_out_channel = width[-2] * 6 # mb7的输出通道×6
            self.last_conv.feature_mix_layer.active_out_channel = width[-1]
    

    # target: 获得网络的参数
    def get_active_subnet_settings(self):
        r = self.active_resolution
        width, depth, kernel_size, expand_ratio= [], [], [],  []

        #first conv
        width.append(self.first_conv.active_out_channel)
        
        #all stage
        for stage_id in range(len(self.block_group_info)): # 遍历全部stage
            start_idx = min(self.block_group_info[stage_id])
            block = self.blocks[start_idx]  #stage中的first block
            width.append(block.mobile_inverted_conv.active_out_channel)
            kernel_size.append(block.mobile_inverted_conv.active_kernel_size)
            expand_ratio.append(block.mobile_inverted_conv.active_expand_ratio)
            depth.append(self.runtime_depth[stage_id])  # 每个stage都是相同的blocks堆叠，体现重复的层数depth/layers就好
        
        # final layer
        if not self.use_v3_head:
            width.append(self.last_conv.active_out_channel)
        else:
            width.append(self.last_conv.feature_mix_layer.active_out_channel)

        return {
            'resolution': r,
            'width': width,
            'kernel_size': kernel_size,
            'expand_ratio': expand_ratio,
            'depth': depth,
        }

    def set_dropout_rate(self, dropout=0, drop_connect=0, drop_connect_only_last_two_stages=True):
        self.active_dropout_rate = dropout
        for idx, block in enumerate(self.blocks):
            if drop_connect_only_last_two_stages:
                if idx not in self.block_group_info[-1] + self.block_group_info[-2]:
                    continue
            this_drop_connect_rate = drop_connect * float(idx) / len(self.blocks)
            block.drop_connect_rate = this_drop_connect_rate


    def sample_min_subnet(self):
        return self._sample_active_subnet(min_net=True)


    def sample_max_subnet(self):
        return self._sample_active_subnet(max_net=True)
    

    def sample_active_subnet(self, compute_flops=False):
        cfg = self._sample_active_subnet(
            False, False
        ) 
        if compute_flops:
            cfg['flops'] = self.compute_active_subnet_flops()
        return cfg
    

    # target：根据flops的上下限采样： 调用采样函数，计算当前采样网络的flops，判断是是否满足flops条件，返回
    def sample_active_subnet_within_range(self, targeted_min_flops, targeted_max_flops):
        while True:
            cfg = self._sample_active_subnet() 
            cfg['flops'] = self.compute_active_subnet_flops()
            if cfg['flops'] >= targeted_min_flops and cfg['flops'] <= targeted_max_flops:
                return cfg

    def _sample_active_subnet(self, min_net=False, max_net=False):

        # 创建一个采样的匿名函数，若非指定最大网络or最小网络，在输入的canidates list中随机采样一个
        sample_cfg = lambda candidates, sample_min, sample_max: \
            min(candidates) if sample_min else (max(candidates) if sample_max else random.choice(candidates))

        cfg = {}
        # sample a resolution
        cfg['resolution'] = sample_cfg(self.cfg_candidates['resolution'], min_net, max_net)
        
        # sample width, depth, kernel_size, expand_ratio
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            cfg[k] = []
            for vv in self.cfg_candidates[k]: # 取width...等候选list中一个数字
                cfg[k].append(sample_cfg(int2list(vv), min_net, max_net))

        # 采样好重设网络参数
        self.set_active_subnet(
            cfg['resolution'], cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio']
        )
        return cfg


    def mutate_and_reset(self, cfg, prob=0.1, keep_resolution=False):
        cfg = copy.deepcopy(cfg)
        
        # 匿名采样函数
        pick_another = lambda x, candidates: x if len(candidates) == 1 else random.choice([v for v in candidates if v != x])
        # sample a resolution
        r = random.random()
        if r < prob and not keep_resolution:
            cfg['resolution'] = pick_another(cfg['resolution'], self.cfg_candidates['resolution'])

        # sample channels, depth, kernel_size, expand_ratio
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            for _i, _v in enumerate(cfg[k]): #遍历 width等list中的元素
                r = random.random()
                if r < prob:
                    cfg[k][_i] = pick_another(cfg[k][_i], int2list(self.cfg_candidates[k][_i])) # cfg[k][_i]是个数，self.cfg_candidates[k][_i]是个list

        self.set_active_subnet(
            cfg['resolution'], cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio']
        )
        return cfg


    # target: 交叉，两个config，根据p两个中挑一个
    def crossover_and_reset(self, cfg1, cfg2, p=0.5): 
        def _cross_helper(g1, g2, prob):
            assert type(g1) == type(g2)
            if isinstance(g1, int):
                return g1 if random.random() < prob else g2
            elif isinstance(g1, list):
                return [v1 if random.random() < prob else v2 for v1, v2 in zip(g1, g2)]
            else:
                raise NotImplementedError

        cfg = {}
        cfg['resolution'] = cfg1['resolution'] if random.random() < p else cfg2['resolution']
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            cfg[k] = _cross_helper(cfg1[k], cfg2[k], p)

        self.set_active_subnet(
            cfg['resolution'], cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio']
        )
        return cfg


    def get_active_subnet(self, preserve_weight=True):
        with torch.no_grad():
            # first layer
            first_conv = self.first_conv.get_active_subnet(3, preserve_weight) # 根据DynamicConvbnActLayer的实例（上述set parm过程以固定参数），返回一个抽样选中的静态的实例

            blocks = []
            input_channel = first_conv.out_channels
            
            # blocks
            for stage_id, block_idx in enumerate(self.block_group_info):    # 遍历stage
                depth = self.runtime_depth[stage_id]                        # runtime_depth存的是各stage的深度
                active_idx = block_idx[:depth]                              # 根据深度取激活的block
                stage_blocks = []
                for idx in active_idx:                                      # 遍历1个stage中的blocks
                    stage_blocks.append(MobileInvertedResidualBlock(
                        self.blocks[idx].mobile_inverted_conv.get_active_subnet(input_channel, preserve_weight),
                        self.blocks[idx].shortcut.get_active_subnet(input_channel, preserve_weight) if self.blocks[idx].shortcut is not None else None
                    ))
                    input_channel = stage_blocks[-1].mobile_inverted_conv.out_channels
                blocks += stage_blocks                                      # append一个stage的blocks

            if not self.use_v3_head:
                last_conv = self.last_conv.get_active_subnet(input_channel, preserve_weight)
                in_features = last_conv.out_channels
            else:
                final_expand_layer = self.last_conv.final_expand_layer.get_active_subnet(input_channel, preserve_weight)
                feature_mix_layer = self.last_conv.feature_mix_layer.get_active_subnet(input_channel*6, preserve_weight)
                in_features = feature_mix_layer.out_channels
                last_conv = nn.Sequential(
                    final_expand_layer,
                    nn.AdaptiveAvgPool2d((1,1)),
                    feature_mix_layer
                )

            classifier = self.classifier.get_active_subnet(in_features, preserve_weight)

            _subnet = AttentiveNasStaticModel(                              # 调用staticmodel创建子网
                first_conv, blocks, last_conv, classifier, self.active_resolution, use_v3_head=self.use_v3_head
            )
            _subnet.set_bn_param(**self.get_bn_param())
            return _subnet


    def get_active_net_config(self):
        raise NotImplementedError


    def compute_active_subnet_flops(self):

        def count_conv(c_in, c_out, size_out, groups, k):
            kernel_ops = k**2
            output_elements = c_out * size_out**2
            ops = c_in * output_elements * kernel_ops / groups   # cin/groups * k^2 * cout * Hout*Wout
            return ops

        def count_linear(c_in, c_out):
            return c_in * c_out

        total_ops = 0

        # first layer
        c_in = 3
        size_out = self.active_resolution // self.first_conv.stride
        c_out = self.first_conv.active_out_channel

        total_ops += count_conv(c_in, c_out, size_out, 1, 3)
        c_in = c_out

        # mb blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                block = self.blocks[idx]
                # 1乘1卷积输出的通道数
                c_middle = make_divisible(round(c_in * block.mobile_inverted_conv.active_expand_ratio), 8)
                # 1*1 conv
                if block.mobile_inverted_conv.inverted_bottleneck is not None:
                    total_ops += count_conv(c_in, c_middle, size_out, 1, 1)
                # dw conv
                stride = 1 if idx > active_idx[0] else block.mobile_inverted_conv.stride
                if size_out % stride == 0:
                    size_out = size_out // stride
                else:
                    size_out = (size_out +1) // stride
                total_ops += count_conv(c_middle, c_middle, size_out, c_middle, block.mobile_inverted_conv.active_kernel_size)
                # 1*1 conv
                c_out = block.mobile_inverted_conv.active_out_channel
                total_ops += count_conv(c_middle, c_out, size_out, 1, 1)
                #se
                if block.mobile_inverted_conv.use_se:
                    num_mid = make_divisible(c_middle // block.mobile_inverted_conv.depth_conv.se.reduction, divisor=8)
                    total_ops += count_conv(c_middle, num_mid, 1, 1, 1) * 2
                # shortcut
                if block.shortcut and c_in != c_out:
                    total_ops += count_conv(c_in, c_out, size_out, 1, 1)
                c_in = c_out

        if not self.use_v3_head:
            c_out = self.last_conv.active_out_channel
            total_ops += count_conv(c_in, c_out, size_out, 1, 1)
        else:
            c_expand = self.last_conv.final_expand_layer.active_out_channel
            c_out = self.last_conv.feature_mix_layer.active_out_channel
            total_ops += count_conv(c_in, c_expand, size_out, 1, 1)
            total_ops += count_conv(c_expand, c_out, 1, 1, 1)

        # n_classes
        total_ops += count_linear(c_out, self.n_classes)
        return total_ops / 1e6


    def load_weights_from_pretrained_models(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
        assert isinstance(checkpoint, dict)
        pretrained_state_dicts = checkpoint['state_dict']
        for k, v in self.state_dict().items():
            name = 'module.' + k if not k.startswith('module') else k
            v.copy_(pretrained_state_dicts[name])

