import math
import sys
import time
from typing import Iterable
import os
import numpy as np

import torch
import torch.nn as nn

import furnace.utils as utils
import torch.nn.functional as F
from tools.plot import plot, plot_image
from tools.calculate import correlation_calculation
from thop import profile


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    similarity = {}

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if it < len(lr_schedule_values):
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr_schedule_values[-1] * param_group[
                        "lr_scale"]  # Use the last value if out of bounds
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    if it < len(wd_schedule_values):
                        param_group["weight_decay"] = wd_schedule_values[it]
                    else:
                        param_group["weight_decay"] = wd_schedule_values[-1]

        samples, fc, bool_masked_pos, h_label = batch

        fc = fc.float().to(device, non_blocking=True)
        samples = samples.float().to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
        h_label = h_label.to(device, non_blocking=True)

        with torch.no_grad():
            bool_masked_pos = bool_masked_pos.unsqueeze(1).to(torch.bool)

        with torch.cuda.amp.autocast():
            outputs = model(samples, h_label, bool_masked_pos=bool_masked_pos.squeeze())
            outputs, loss_all, sim = outputs

            b, c, seq_len, nvars = samples.size()
            outputs = outputs.reshape(b, -1, seq_len, seq_len)

            loss_NCM, loss_age, loss_RCM = loss_all
            loss = loss_NCM + loss_age + loss_RCM

        loss_value = loss.item()
        loss_NCM_value = loss_NCM.item()
        loss_RCM_value = loss_RCM.item()
        loss_age_value = loss_age.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_NCM=loss_NCM_value)
        metric_logger.update(loss_RCM=loss_RCM_value)
        metric_logger.update(loss_age_value=loss_age_value)
        metric_logger.update(loss_scale=loss_scale_value)


        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss=loss_NCM_value, head="loss_NCM")
            log_writer.update(loss=loss_RCM_value, head="loss_RCM")
            log_writer.update(loss=loss_age_value, head="loss_age")

            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    metric_logger.synchronize_between_processes()
    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(now_time, "Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, similarity
