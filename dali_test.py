# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys
from datetime import date

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import models
from utils.config import setup
from utils.flops_counter import count_net_flops_and_params
import utils.comm as comm
import utils.saver as saver

from data.data_loader import build_data_loader
from utils.progress import AverageMeter, ProgressMeter, accuracy
import argparse

# dali import
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

parser = argparse.ArgumentParser(description='Test AttentiveNas Models')
parser.add_argument('--config-file', default='./configs/eval_attentive_nas_models.yml')
parser.add_argument('--model', default='a0', type=str, choices=['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
parser.add_argument('--gpu', default=0, type=int, help='gpu id')


run_args = parser.parse_args()

if __name__ == '__main__':
    
    start = time.process_time()

    args = setup(run_args.config_file)
    args.model = run_args.model
    args.gpu = run_args.gpu
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #cudnn.benchmark = True

    args.__dict__['active_subnet'] = args.__dict__['pareto_models'][args.model]                 # 根据args.model是a0读取模型参数后赋值args的active_subnet条目
    print(args.active_subnet)

    # dali dataloader

    # train_loader, val_loader, train_sampler = build_data_loader(args)

    args.world_size = 1
    args.local_rank = 0
    args.dali_cpu = False

    @pipeline_def
    def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
        images, labels = fn.readers.file(file_root=data_dir,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        random_shuffle=is_training,
                                        pad_last_batch=False,
                                        name="Reader")
        
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        if is_training:
            images = fn.decoders.image_random_crop(images,                                      # output:HWC
                                                device=decoder_device, output_type=types.RGB,
                                                device_memory_padding=device_memory_padding,
                                                host_memory_padding=host_memory_padding,
                                                random_aspect_ratio=[3. / 4., 4. / 3.],
                                                random_area=[0.08, 1.0])
            images = fn.resize(images,
                            device=dali_device,
                            resize_x=crop,
                            resize_y=crop,
                            interp_type=types.INTERP_LINEAR)
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.decoders.image(images,
                                    device=decoder_device,
                                    output_type=types.RGB)
            images = fn.resize(images,
                            device=dali_device,
                            size=size,
                            mode="not_smaller",
                            interp_type=types.INTERP_LINEAR)
            mirror = False

        images = fn.crop_mirror_normalize(images.gpu(),
                                        dtype=types.FLOAT,
                                        output_layout="CHW",
                                        crop=(crop, crop),
                                        mean=[0.485 * 255,0.456 * 255,0.406 * 255],             #这里由于未将原始图片从0-255转到0-1，所以均值和方差要乘上255
                                        std=[0.229 * 255,0.224 * 255,0.225 * 255],                                    
                                        mirror=mirror)
        labels = labels.gpu()
        return images, labels
    
    traindir = os.path.join(args.dataset_dir, "train")
    valdir = os.path.join(args.dataset_dir, "val")

    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.data_loader_workers_per_gpu,
                                device_id=args.local_rank,
                                data_dir=traindir,
                                crop=getattr(args, 'train_crop_size', 224),
                                size=getattr(args, 'test_scale', 256),
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=True)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP)

    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.data_loader_workers_per_gpu,
                                device_id=args.local_rank,
                                data_dir=valdir,
                                crop=getattr(args, 'train_crop_size', 224),
                                size=getattr(args, 'test_scale', 256),
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=False)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

                
    ## init static attentivenas model with weights inherited from the supernet 
    model = models.model_factory.create_model(args)

    model.to(args.gpu)
    model.eval()
    
    # bn running stats calibration following Slimmable (https://arxiv.org/abs/1903.05134)
    # please consider trying a different random seed if you see a small accuracy drop
    with torch.no_grad():
        model.reset_running_stats_for_calibration()                         # 清零BN的数据（平均=0，方差=1，追踪的batch数量=0）

        for batch_idx, data_list in enumerate(train_loader):
            images = data_list[0]['data']
            if batch_idx >= args.post_bn_calibration_batch_num:             # 仅前向指定数量次，计算bn
                break
            #images = images.cuda(args.gpu, non_blocking=True)              # 图像已经通过dali在gpu上处理了
            model(images)  #forward only
    
    model.eval()                                                            # 固定训练时bn的数量
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss().cuda()

        from evaluate.imagenet_eval import dali_validate_one_subnet
        acc1, acc5, loss, flops, params = dali_validate_one_subnet(val_loader, model, criterion, args)
        print(acc1, acc5, flops, params)
    end = time.process_time()
    print(end - start)


