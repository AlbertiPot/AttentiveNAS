import os
# dali import
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

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
                                    mean=[0.485 * 255,0.456 * 255,0.406 * 255],             # 这里由于未将原始图片从0-255转到0-1，所以均值和方差要乘上255
                                    std=[0.229 * 255,0.224 * 255,0.225 * 255],                                    
                                    mirror=mirror)
    labels = labels.gpu()
    return images, labels
    
    

def build_dali_data_loader(args):

    traindir = os.path.join(args.dataset_dir, "train")
    valdir = os.path.join(args.dataset_dir, "val")

    pipetrain = create_dali_pipeline(batch_size=args.batch_size,                                 # batch_size, num_threads, device_id是decorator@pipeline_def的参数
                                num_threads=args.data_loader_workers_per_gpu,
                                device_id=args.local_rank,
                                data_dir=traindir,
                                crop=getattr(args, 'train_crop_size', 224),
                                size=getattr(args, 'test_scale', 256),
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=True)
    pipetrain.build()
    train_loader = DALIClassificationIterator(pipetrain, reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP)

    pipeval = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.data_loader_workers_per_gpu,
                                device_id=args.local_rank,
                                data_dir=valdir,
                                crop=getattr(args, 'train_crop_size', 224),
                                size=getattr(args, 'test_scale', 256),
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=False)
    pipeval.build()
    val_loader = DALIClassificationIterator(pipeval, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    return train_loader, val_loader