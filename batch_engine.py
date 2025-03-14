import math
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tools.distributed import reduce_tensor
from tools.utils import AverageMeter, to_scalar, time_str
from torch.cuda.amp import GradScaler, autocast  # 新增：导入混合精度训练所需模块

scaler = GradScaler()

def logits4pred(criterion, logits_list):
    if criterion.__class__.__name__.lower() in ['bceloss']:
        logits = logits_list[0]
        probs = logits.sigmoid()
    else:
        assert False, f"{criterion.__class__.__name__.lower()} not exits"

    return probs, logits


# def batch_trainer(cfg, args, epoch, model, model_ema, train_loader, criterion, optimizer, loss_w=[1, ], scheduler=None):
#     model.train()
#     epoch_time = time.time()
#
#     loss_meter = AverageMeter()
#     subloss_meters = [AverageMeter() for i in range(len(loss_w))]
#
#     batch_num = len(train_loader)
#     gt_list = []
#     preds_probs = []
#     preds_logits = []
#     imgname_list = []
#     loss_mtr_list = []
#
#     lr = optimizer.param_groups[1]['lr']
#
#     # 新增：初始化梯度累积步数和混合精度训练的梯度缩放器
#     GRADIENT_ACCUMULATION_STEPS = 2  # 累积步数
#     scaler = GradScaler()
#
#     for step, (imgs, gt_label, imgname) in enumerate(train_loader):
#         iter_num = epoch * len(train_loader) + step
#
#         batch_time = time.time()
#         imgs, gt_label = imgs.cuda(), gt_label.cuda()
#
#         # 新增：使用混合精度训练
#         # with autocast():
#         #     train_logits, feat = model(imgs, gt_label)
#         #     loss_list, loss_mtr = criterion(train_logits, gt_label)
#         #
#         #     train_loss = 0
#         #     for i, l in enumerate(loss_w):
#         #         train_loss += loss_list[i] * l
#
#         # 暂时关闭混合精度训练
#         # with autocast():  # 注释掉这一行
#         train_logits, feat = model(imgs, gt_label)
#         loss_list, loss_mtr = criterion(train_logits, gt_label)
#
#         train_loss = 0
#         for i, l in enumerate(loss_w):
#             train_loss += loss_list[i] * l
#
#         # train_loss = train_loss / GRADIENT_ACCUMULATION_STEPS  # 注释掉这一行
#         train_loss.backward()  # 直接进行反向传播
#
#         train_loss = train_loss / GRADIENT_ACCUMULATION_STEPS  # 平均损失
#
#         # 新增：使用混合精度训练的梯度缩放
#         scaler.scale(train_loss).backward()
#
#         # 新增：梯度累积
#         if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
#             if cfg.TRAIN.CLIP_GRAD:
#                 scaler.unscale_(optimizer)  # 新增：在梯度裁剪前取消缩放
#                 # clip_grad_norm_(model.parameters(), max_norm=10.0)  # make larger learning rate works
#                 clip_grad_norm_(model.parameters(), max_norm=1.0)  # make larger learning rate works
#             scaler.step(optimizer)  # 新增：使用缩放器更新参数
#             scaler.update()  # 新增：更新缩放器
#             optimizer.zero_grad()
#
#         if cfg.TRAIN.LR_SCHEDULER.TYPE == 'annealing_cosine' and scheduler is not None:
#             if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:  # 新增：在梯度累积步数达到时更新学习率
#                 scheduler.step()
#
#         if model_ema is not None:
#             if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:  # 新增：在梯度累积步数达到时更新EMA模型
#                 model_ema.update(model)
#
#         torch.cuda.synchronize()
#         torch.cuda.empty_cache()  # 新增：释放未使用的缓存显存
#
#         if len(loss_list) > 1:
#             for i, meter in enumerate(subloss_meters):
#                 meter.update(
#                     to_scalar(reduce_tensor(loss_list[i], args.world_size)
#                               if args.distributed else loss_list[i]))
#         loss_meter.update(to_scalar(reduce_tensor(train_loss, args.world_size) if args.distributed else train_loss))
#
#         train_probs, train_logits = logits4pred(criterion, train_logits)
#
#         gt_list.append(gt_label.cpu().numpy())
#         preds_probs.append(train_probs.detach().cpu().numpy())
#         preds_logits.append(train_logits.detach().cpu().numpy())
#
#         imgname_list.append(imgname)
#
#         log_interval = 1
#
#         if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
#             if args.local_rank == 0:
#                 print(f'{time_str()}, '
#                       f'Step {step}/{batch_num} in Ep {epoch}, '
#                       f'LR: [{optimizer.param_groups[0]["lr"]:.1e}, {optimizer.param_groups[1]["lr"]:.1e}] '
#                       f'Time: {time.time() - batch_time:.2f}s , '
#                       f'train_loss: {loss_meter.avg:.4f}, ')
#
#                 print([f'{meter.avg:.4f}' for meter in subloss_meters])
#
#             # break
#
#     train_loss = loss_meter.avg
#
#     gt_label = np.concatenate(gt_list, axis=0)
#     preds_probs = np.concatenate(preds_probs, axis=0)
#
#     if args.local_rank == 0:
#         print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')
#
#     return train_loss, gt_label, preds_probs, imgname_list, preds_logits, loss_mtr_list
def batch_trainer(cfg, args, epoch, model, model_ema, train_loader, criterion, optimizer, loss_w=[1, ], scheduler=None):
    model.train()
    epoch_time = time.time()

    loss_meter = AverageMeter()
    subloss_meters = [AverageMeter() for i in range(len(loss_w))]

    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []
    preds_logits = []
    imgname_list = []
    loss_mtr_list = []

    lr = optimizer.param_groups[1]['lr']

    for step, (imgs, gt_label, imgname) in enumerate(train_loader):
        iter_num = epoch * len(train_loader) + step

        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        train_logits, feat = model(imgs, gt_label)

        loss_list, loss_mtr = criterion(train_logits, gt_label)

        train_loss = 0

        for i, l in enumerate(loss_w):
            train_loss += loss_list[i] * l

        optimizer.zero_grad()
        # 假设 scaler 已经定义
        scaler = torch.cuda.amp.GradScaler()
        scaler.scale(train_loss).backward()

        if cfg.TRAIN.CLIP_GRAD:
            clip_grad_norm_(model.parameters(), max_norm=10.0)  # make larger learning rate works

        scaler.step(optimizer)
        scaler.update()

        if cfg.TRAIN.LR_SCHEDULER.TYPE == 'annealing_cosine' and scheduler is not None:
            scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()

        if len(loss_list) > 1:
            for i, meter in enumerate(subloss_meters):
                meter.update(
                    to_scalar(reduce_tensor(loss_list[i], args.world_size)
                              if args.distributed else loss_list[i]))
        loss_meter.update(to_scalar(reduce_tensor(train_loss, args.world_size) if args.distributed else train_loss))

        train_probs, train_logits = logits4pred(criterion, train_logits)

        gt_list.append(gt_label.cpu().numpy())
        preds_probs.append(train_probs.detach().cpu().numpy())
        preds_logits.append(train_logits.detach().cpu().numpy())

        imgname_list.append(imgname)

        log_interval = 1

        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            if args.local_rank == 0:
                print(f'{time_str()}, '
                      f'Step {step}/{batch_num} in Ep {epoch}, '
                      f'LR: [{optimizer.param_groups[0]["lr"]:.1e}, {optimizer.param_groups[1]["lr"]:.1e}] '
                      f'Time: {time.time() - batch_time:.2f}s , '
                      f'train_loss: {loss_meter.avg:.4f}, ')

                print([f'{meter.avg:.4f}' for meter in subloss_meters])

            # break

    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    if args.local_rank == 0:
        print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs, imgname_list, preds_logits, loss_mtr_list


def valid_trainer(cfg, args, epoch, model, valid_loader, criterion, loss_w=[1, ]):
    model.eval()
    loss_meter = AverageMeter()
    subloss_meters = [AverageMeter() for i in range(len(loss_w))]

    preds_probs = []
    preds_logits = []
    gt_list = []
    imgname_list = []
    loss_mtr_list = []

    with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(tqdm(valid_loader)):
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            valid_logits, feat = model(imgs, gt_label)


            loss_list, loss_mtr = criterion(valid_logits, gt_label)
            valid_loss = 0
            for i, l in enumerate(loss_list):
                valid_loss += loss_w[i] * l

            valid_probs, valid_logits = logits4pred(criterion, valid_logits)
            preds_probs.append(valid_probs.cpu().numpy())
            preds_logits.append(valid_logits.cpu().numpy())

            if len(loss_list) > 1:
                for i, meter in enumerate(subloss_meters):
                    meter.update(
                        to_scalar(reduce_tensor(loss_list[i], args.world_size) if args.distributed else loss_list[i]))
            loss_meter.update(to_scalar(reduce_tensor(valid_loss, args.world_size) if args.distributed else valid_loss))

            torch.cuda.synchronize()

            imgname_list.append(imgname)

    valid_loss = loss_meter.avg

    if args.local_rank == 0:
        print([f'{meter.avg:.4f}' for meter in subloss_meters])

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    preds_logits = np.concatenate(preds_logits, axis=0)

    return valid_loss, gt_label, preds_probs, imgname_list, preds_logits, loss_mtr_list
