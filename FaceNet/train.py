import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.facenet import Facenet
from nets.facenet_training import (get_lr_scheduler, set_optimizer_lr,
                                   triplet_loss, weights_init)
from utils.callback import LossHistory
from utils.dataloader import FacenetDataset, LFWDataset, dataset_collate
from utils.utils import get_num_classes, show_config
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    Cuda            = True
    distributed     = False
    sync_bn         = False
    fp16            = False
    annotation_path = "cls_train.txt"
    input_shape     = [160, 160, 3]
    backbone        = "mobilenet"
    model_path      = "model_data/facenet_mobilenet.pth"
    pretrained      = False
    batch_size      = 96
    Init_Epoch      = 0
    Epoch           = 100
    Init_lr             = 1e-3
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    lr_decay_type       = "cos"
    save_period         = 1
    save_dir            = 'logs'
    num_workers     = 4
    lfw_eval_flag   = True
    lfw_dir_path    = "lfw"
    lfw_pairs_path  = "model_data/lfw_pair.txt"
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    num_classes = get_num_classes(annotation_path)
    model = Facenet(backbone=backbone, num_classes=num_classes, pretrained=pretrained)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    loss            = triplet_loss()

    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:

            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    LFW_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=32, shuffle=False) if lfw_eval_flag else None


    val_split = 0.01
    with open(annotation_path,"r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    show_config(
        num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )

    if True:
        if batch_size % 3 != 0:
            raise ValueError("Batch_size must be the multiple of 3.")
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        train_dataset   = FacenetDataset(input_shape, lines[:num_train], num_classes, random = True)
        val_dataset     = FacenetDataset(input_shape, lines[num_train:], num_classes, random = False)
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True
        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size//3, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size//3, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)
        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, Cuda, LFW_loader, batch_size//3, lfw_eval_flag, fp16, scaler, save_period, save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()
