# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import atexit
import os
import random
import copy

def count_helper(v, flops, m):
    if flops not in m:                                                                  # 若flops不在prob_map['resolution']中，创建一个新的flops的计数字典
        m[flops] = {}
    if v not in m[flops]:                                                               # 若prob_map['resolution'][flops]下无对应的resolution，创建一个新的计数字典prob_map['resolution'][flops][resolution] = 0
        m[flops][v] = 0
    m[flops][v] += 1 


def round_flops(flops, step):
    return int(round(flops / step) * step)


def convert_count_to_prob(m):
    if isinstance(m[list(m.keys())[0]], dict):
        for k in m:
            convert_count_to_prob(m[k])
    else:
        t = sum(m.values())
        for k in m:
            m[k] = 1.0 * m[k] / t                                                       # 频数做概率


def sample_helper(flops, m):
    keys = list(m[flops].keys())
    probs = list(m[flops].values())
    return random.choices(keys, weights=probs)[0]


def build_trasition_prob_matrix(file_handler, step):
    # initlizie
    prob_map = {}
    prob_map['discretize_step'] = step
    for k in ['flops', 'resolution', 'width', 'depth', 'kernel_size', 'expand_ratio']:
        prob_map[k] = {}

    cc = 0                                                                              # 模型计数
    
    # 每一行是一个模型，计算该模型的flops为先验条件下，
    for line in file_handler:
        vals = eval(line.strip())                                                       # .strip()去掉头部尾部的空格或；eval是执行字符串表达式, 这里是将line这一行字符串变为一个字典值赋给vals

        # discretize
        flops = round_flops(vals['flops'], step)                                        # 离散化，对flops取整赋给prob_map
        prob_map['flops'][flops] = prob_map['flops'].get(flops, 0) + 1                  # 对该flops技术加1，get(key, default)，若指定key没有值，则get返回default    

        # resolution
        r = vals['resolution']
        count_helper(r, flops, prob_map['resolution'])

        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            for idx, v in enumerate(vals[k]):                                           # idx 是层数索引，v是对应的width,depth,ks,expand_ration等值
                if idx not in prob_map[k]:
                    prob_map[k][idx] = {}
                count_helper(v, flops, prob_map[k][idx])

        cc += 1

    # convert count to probability
    for k in ['flops', 'resolution', 'width', 'depth', 'kernel_size', 'expand_ratio']:
        convert_count_to_prob(prob_map[k])
    prob_map['n_observations'] = cc
    return prob_map



class ArchSampler():
    def __init__(self, arch_to_flops_map_file_path, discretize_step, model, acc_predictor=None):
        super(ArchSampler, self).__init__()
        with open(arch_to_flops_map_file_path, 'r') as fp:
            self.prob_map = build_trasition_prob_matrix(fp, discretize_step)            # 调用以上函数构建一个多层dict，存储采样各个flops下解析度等的概率

        self.discretize_step = discretize_step
        self.model = model

        self.acc_predictor = acc_predictor

        self.min_flops = min(list(self.prob_map['flops'].keys()))
        self.max_flops = max(list(self.prob_map['flops'].keys()))

        self.curr_sample_pool = None #TODO; architecture samples could be generated in an asynchronous way


    def sample_one_target_flops(self, flops_uniform=False):
        f_vals = list(self.prob_map['flops'].keys())
        f_probs = list(self.prob_map['flops'].values())

        if flops_uniform:
            return random.choice(f_vals)                                                # 均匀选择，https://docs.python.org/3/library/random.html
        else:
            return random.choices(f_vals, weights=f_probs)[0]                           # 有放回采样，权重是概率, [0]是从list中取一个val


    def sample_archs_according_to_flops(self, target_flops,  n_samples=1, max_trials=100, return_flops=True, return_trials=False):
        archs = []
        #for _ in range(n_samples):
        while len(archs) < n_samples:
            for _trial in range(max_trials+1):
                arch = {}
                arch['resolution'] = sample_helper(target_flops, self.prob_map['resolution'])   # sample_helper执行按照权重采样，在resolution字典中以flops为先验，根据resolution的频次为权重采样
                for k in ['width', 'kernel_size', 'depth', 'expand_ratio']:                     # 对每层的width，kernel_size等采样
                    arch[k] = []
                    for idx in sorted(list(self.prob_map[k].keys())):
                        arch[k].append(sample_helper(target_flops, self.prob_map[k][idx]))
                if self.model:
                    self.model.set_active_subnet(**arch)
                    flops = self.model.compute_active_subnet_flops()
                    if return_flops:
                        arch['flops'] = flops
                    if round_flops(flops, self.discretize_step) == target_flops:
                        break
                else:
                    raise NotImplementedError
            #accepte the sample anyway 若采样到最后一个仍未符合target flops，则接纳最后一个采样
            archs.append(arch)
        return archs    


