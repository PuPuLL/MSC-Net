import torch
from scipy.io import savemat
import numpy as np
from timm.utils import accuracy
import furnace.utils as utils
from calculate import sencitivity_specificity
from sklearn.manifold import TSNE
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def test(data_loader, model, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    mf_list, label_list = [], []
    embeddings_list = []
    output_list, attn_list = [], []

    model.eval()
    softmax = torch.nn.Softmax(dim=1).to(device)

    for batch in metric_logger.log_every(data_loader, 10, header):
        ts = batch[0]
        target = batch[-1]

        target = target.to(device, non_blocking=True)
        label_list.append(target)

        ts = ts.float().to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(ts)
            output_list.append(output)

            preds = softmax(output)
            _, pred_small = preds.topk(1, dim=1, largest=True, sorted=True)
            mf_list.append(pred_small.squeeze())  
            embeddings_list.append(output.cpu().numpy())

    mf = torch.cat(mf_list).view(-1, 1) 
    label = torch.cat(label_list).view(-1)
    outputs = torch.cat(output_list)

    all_outputs = [torch.zeros_like(outputs) for _ in range(world_size)]
    dist.all_gather(all_outputs, outputs)
    global_outputs = torch.cat(all_outputs)

    # 1. 收集所有卡的预测结果
    all_preds = [torch.zeros_like(mf) for _ in range(world_size)]
    dist.all_gather(all_preds, mf)
    global_preds = torch.cat(all_preds)

    # 2. 收集所有卡的真实标签
    all_labels = [torch.zeros_like(label) for _ in range(world_size)]
    dist.all_gather(all_labels, label)
    global_labels = torch.cat(all_labels)

    # 4. 收集所有卡的attention
    all_attns = [torch.zeros_like(attns) for _ in range(world_size)]
    dist.all_gather(all_attns, attns)
    global_attns = torch.cat(all_attns)

    # ===== 只在主卡计算全局指标 =====
    if rank == 0:
        # 计算全局准确率
        global_acc = accuracy(global_outputs, global_labels, topk=(1,))[0].item()

        # 计算全局敏感度/特异度/AUC

        global_sensitivity, global_specificity, global_auc = sencitivity_specificity(
            global_preds, global_labels
        )

    else:
        # 非主卡返回空值
        global_acc, global_sensitivity, global_specificity, global_auc = 0, 0, 0, 0

    # 广播全局结果到所有卡
    if dist.is_initialized():
        # 将指标打包成tensor广播
        metrics_tensor = torch.tensor([
            global_acc, global_sensitivity, global_specificity, global_auc
        ]).float().to(device)
        dist.broadcast(metrics_tensor, src=0)

        # 解包广播结果
        global_acc, global_sensitivity, global_specificity, global_auc = metrics_tensor.cpu().numpy()

    # 更新metric_logger
    metric_logger.meters["test_acc"].update(global_acc)

    # 返回全局结果
    return global_acc, global_sensitivity * 100, global_specificity * 100, \
    global_auc, global_labels, global_preds, global_attns
