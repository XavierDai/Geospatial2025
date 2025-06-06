#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lpbert_ddp.py

完整实现 LP-BERT 的分布式训练脚本，支持：
- 智能缓存：先单GPU预处理数据，后多GPU训练
- 断点续传：自动找到最新的checkpoint继续训练
- 周期性Geo-BLEU评估
- 学习率动态调整（ReduceLROnPlateau + Warmup）
- 在线随机Mask（每个batch重新生成）
- 可配置的模型架构参数
- 训练日志记录（txt和csv格式）
- 混合精度训练支持
- LP-BERT 模型（Embedding + TransformerEncoder + 预测头）
- 单GPU/多GPU DDP 训练

使用方法：
1. 先运行单GPU预处理：
   python train_lpbert_ddp.py --raw_csv_path ./yjmob100k-dataset1.csv --cache_dir ./cache_lpbert --batch_size 4 --epochs 1

2. 再运行多GPU训练（小模型）：
   torchrun --nproc_per_node=4 --master_port=12345 train_lpbert_ddp.py \
     --raw_csv_path ./yjmob100k-dataset1.csv \
     --cache_dir ./cache_lpbert \
     --batch_size 32 \
     --epochs 200 \
     --lr 5e-4

3. 大模型训练：
   torchrun --nproc_per_node=4 --master_port=12345 train_lpbert_ddp.py \
     --raw_csv_path ./yjmob100k-dataset1.csv \
     --cache_dir ./cache_lpbert_large \
     --batch_size 16 \
     --epochs 200 \
     --lr 1e-4 \
     --emb_dim 256 \
     --num_layers 6 \
     --nhead 16 \
     --dim_feedforward 1024 \
     --warmup_steps 1000

4. 断点续传：
   添加 --resume 参数
"""

import os
import sys
import argparse
import pickle
import random
import re
import csv
import json
from collections import defaultdict, Counter
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

# ------------------------------
# 1. 参数解析
# ------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LP-BERT Distributed Training Script")

    # 数据相关参数
    parser.add_argument(
        "--raw_csv_path", type=str, default="./yjmob100k-dataset1.csv",
        help="原始 CSV 数据路径"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./cache_lpbert",
        help="预处理后数据的缓存目录"
    )
    parser.add_argument(
        "--mask_days", type=int, default=15,
        help="对训练集中每个用户随机连续 Mask 的天数"
    )
    
    # 训练相关参数
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="全局 Batch Size（DDP 会分摊到每张卡上）"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="训练轮数"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4,
        help="学习率"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0,
        help="Warmup步数，0表示不使用warmup"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="是否从最新的checkpoint恢复训练"
    )
    parser.add_argument(
        "--use_amp", action="store_true",
        help="是否使用自动混合精度训练"
    )
    
    # 模型架构参数
    parser.add_argument(
        "--emb_dim", type=int, default=128,
        help="Embedding维度"
    )
    parser.add_argument(
        "--num_layers", type=int, default=4,
        help="Transformer层数"
    )
    parser.add_argument(
        "--nhead", type=int, default=8,
        help="注意力头数"
    )
    parser.add_argument(
        "--dim_feedforward", type=int, default=512,
        help="前馈层维度"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout率"
    )
    parser.add_argument(
        "--max_pos_emb", type=int, default=4096,
        help="Position Embedding 的最大长度"
    )
    
    # 评估相关参数
    parser.add_argument(
        "--eval_interval", type=int, default=100,
        help="每多少个batch计算一次Geo-BLEU"
    )
    parser.add_argument(
        "--geo_bleu_n", type=int, default=4,
        help="Geo-BLEU的最大n-gram"
    )
    
    # 学习率调度器参数
    parser.add_argument(
        "--scheduler_patience", type=int, default=10,
        help="学习率调度器的patience"
    )
    parser.add_argument(
        "--scheduler_factor", type=float, default=0.5,
        help="学习率调度器的下降因子"
    )
    parser.add_argument(
        "--scheduler_threshold", type=float, default=0.01,
        help="学习率调度器的改善阈值"
    )

    return parser.parse_args()

# ------------------------------
# 2. Geo-BLEU 计算
# ------------------------------

def extract_ngrams(sequence, n):
    """提取序列的n-gram"""
    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i+n])
        ngrams.append(ngram)
    return ngrams

def compute_geo_bleu(pred_sequences, target_sequences, max_n=4):
    """
    计算Geo-BLEU分数
    pred_sequences: list of predicted location sequences
    target_sequences: list of target location sequences
    max_n: 最大n-gram (通常为4)
    """
    if len(pred_sequences) == 0 or len(target_sequences) == 0:
        return 0.0
    
    bleu_scores = []
    
    for pred_seq, target_seq in zip(pred_sequences, target_sequences):
        if len(pred_seq) == 0 or len(target_seq) == 0:
            bleu_scores.append(0.0)
            continue
            
        # 计算不同n-gram的精度
        precisions = []
        
        for n in range(1, min(max_n + 1, len(pred_seq) + 1)):
            pred_ngrams = extract_ngrams(pred_seq, n)
            target_ngrams = extract_ngrams(target_seq, n)
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            # 计算n-gram匹配数
            target_ngram_counts = Counter(target_ngrams)
            matches = 0
            
            for ngram in pred_ngrams:
                if target_ngram_counts[ngram] > 0:
                    matches += 1
                    target_ngram_counts[ngram] -= 1
            
            precision = matches / len(pred_ngrams) if len(pred_ngrams) > 0 else 0.0
            precisions.append(precision)
        
        # 计算几何平均
        if precisions and all(p > 0 for p in precisions):
            log_precision = sum(np.log(p) for p in precisions) / len(precisions)
            geo_mean = np.exp(log_precision)
        else:
            geo_mean = 0.0
        
        # 简化版本：不使用brevity penalty
        bleu_scores.append(geo_mean)
    
    return np.mean(bleu_scores) if bleu_scores else 0.0

def evaluate_batch_geo_bleu(model, batch_data, device, num_locations, max_n=4):
    """在一个batch上评估Geo-BLEU"""
    model.eval()
    
    date_tensor, time_tensor, loc_tensor, delta_tensor, padding_mask, token_mask, target_locations = batch_data
    
    # 找到被mask的位置
    masked_indices = token_mask.nonzero(as_tuple=False)
    if masked_indices.size(0) == 0:
        return 0.0
    
    with torch.no_grad():
        # 获取模型预测
        logits_masked = model(date_tensor, time_tensor, loc_tensor, delta_tensor, padding_mask, token_mask)
        predictions = torch.argmax(logits_masked, dim=-1)
    
    # 整理预测和目标序列
    pred_sequences = []
    target_sequences = []
    
    # 按用户分组
    batch_predictions = defaultdict(list)
    batch_targets = defaultdict(list)
    
    for idx, (batch_idx, seq_idx) in enumerate(masked_indices):
        batch_idx = batch_idx.item()
        seq_idx = seq_idx.item()
        
        pred_loc = predictions[idx].item()
        target_loc = target_locations[batch_idx, seq_idx].item()
        
        if target_loc >= 0:  # 有效的目标
            batch_predictions[batch_idx].append(pred_loc)
            batch_targets[batch_idx].append(target_loc)
    
    # 转换为列表
    for batch_idx in batch_predictions:
        if batch_predictions[batch_idx] and batch_targets[batch_idx]:
            pred_sequences.append(batch_predictions[batch_idx])
            target_sequences.append(batch_targets[batch_idx])
    
    # 计算Geo-BLEU
    geo_bleu = compute_geo_bleu(pred_sequences, target_sequences, max_n)
    
    model.train()
    return geo_bleu

# ------------------------------
# 3. Checkpoint 管理
# ------------------------------

def find_latest_checkpoint(cache_dir):
    """在cache_dir中找到最新的checkpoint文件"""
    if not os.path.exists(cache_dir):
        return None, 0
    
    # 查找所有checkpoint文件
    checkpoint_files = []
    pattern = re.compile(r'checkpoint_epoch(\d+)\.pt')
    
    for filename in os.listdir(cache_dir):
        match = pattern.match(filename)
        if match:
            epoch_num = int(match.group(1))
            checkpoint_files.append((filename, epoch_num))
    
    if not checkpoint_files:
        return None, 0
    
    # 找到epoch最大的checkpoint
    checkpoint_files.sort(key=lambda x: x[1])
    latest_file, latest_epoch = checkpoint_files[-1]
    
    return os.path.join(cache_dir, latest_file), latest_epoch

def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, device):
    """加载checkpoint"""
    if device.type == 'cuda':
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 处理DDP保存的模型
    state_dict = checkpoint['model_state_dict']
    # 如果是DDP模型保存的，去掉module.前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载scheduler状态（如果有）
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 加载scaler状态（如果有）
    if 'scaler_state_dict' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('loss', 0.0)

# ------------------------------
# 4. 训练日志管理
# ------------------------------

def log_training_info(cache_dir, epoch, avg_loss, avg_geo_bleu, learning_rate, extra_info=None):
    """记录训练信息到多个格式的文件"""
    # 1. 简单文本日志
    log_path = os.path.join(cache_dir, "training_log.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        log_line = f"[{timestamp}] Epoch {epoch}: loss={avg_loss:.6f}, geo_bleu={avg_geo_bleu:.6f}, lr={learning_rate:.2e}"
        if extra_info:
            log_line += f", {extra_info}"
        f.write(log_line + "\n")
    
    # 2. CSV格式日志
    csv_log_path = os.path.join(cache_dir, "training_metrics.csv")
    file_exists = os.path.isfile(csv_log_path)
    with open(csv_log_path, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'avg_loss', 'avg_geo_bleu', 'learning_rate', 'timestamp'])
        writer.writerow([epoch, avg_loss, avg_geo_bleu, learning_rate, timestamp])
    
    # 3. JSON格式日志（便于程序读取）
    json_log_path = os.path.join(cache_dir, "training_history.json")
    try:
        with open(json_log_path, "r") as f:
            history = json.load(f)
    except:
        history = []
    
    history.append({
        'epoch': epoch,
        'avg_loss': avg_loss,
        'avg_geo_bleu': avg_geo_bleu,
        'learning_rate': learning_rate,
        'timestamp': timestamp
    })
    
    with open(json_log_path, "w") as f:
        json.dump(history, f, indent=2)

# ------------------------------
# 5. 数据预处理与缓存
# ------------------------------

def load_raw_data(csv_path):
    """读取原始 CSV 数据"""
    print(f"[INFO] 正在读取 CSV 文件: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] CSV 文件读取完成，数据量: {len(df)} 行")
    
    # 检查必要的列
    expected_cols = {"uid", "d", "t", "x", "y"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV 缺少必要列，应该包含：{expected_cols}")
    return df

def build_user_sequences(df):
    """构建用户序列，计算 location_id 和时间间隔"""
    print(f"[INFO] 开始构建用户序列，总数据量: {len(df)}")
    
    # 首先检查数据范围
    print(f"[INFO] 数据范围检查:")
    print(f"  uid: {df['uid'].min()} - {df['uid'].max()}")
    print(f"  d: {df['d'].min()} - {df['d'].max()}")  
    print(f"  t: {df['t'].min()} - {df['t'].max()}")
    print(f"  x: {df['x'].min()} - {df['x'].max()}")
    print(f"  y: {df['y'].min()} - {df['y'].max()}")
    
    grid_width = 200
    
    # 使用向量化操作提高效率
    df_copy = df.copy()
    
    # 修正：将1-based索引转换为0-based索引
    df_copy['d_0based'] = df_copy['d'] - 1  # 1-75 转换为 0-74
    df_copy['t_0based'] = df_copy['t']      # 0-47 保持不变
    df_copy['x_0based'] = df_copy['x'] - 1  # 1-200 转换为 0-199
    df_copy['y_0based'] = df_copy['y'] - 1  # 1-200 转换为 0-199
    
    # 重新计算 loc_id (基于0-based索引)
    df_copy['loc_id'] = df_copy['x_0based'] * grid_width + df_copy['y_0based']
    df_copy['abs_idx'] = df_copy['d_0based'] * 48 + df_copy['t_0based']
    
    print(f"[INFO] 转换后的数据范围:")
    print(f"  d_0based: {df_copy['d_0based'].min()} - {df_copy['d_0based'].max()}")
    print(f"  t_0based: {df_copy['t_0based'].min()} - {df_copy['t_0based'].max()}")
    print(f"  loc_id: {df_copy['loc_id'].min()} - {df_copy['loc_id'].max()}")
    
    print("[INFO] 按用户分组数据...")
    grouped = df_copy.groupby('uid')
    
    user2seq = {}
    total_users = len(grouped)
    
    print(f"[INFO] 处理 {total_users} 个用户的序列...")
    
    for i, (uid, group) in enumerate(tqdm(grouped, desc="构建用户序列")):
        # 按绝对时间索引排序
        group_sorted = group.sort_values('abs_idx')
        
        # 计算时间间隔
        abs_indices = group_sorted['abs_idx'].values
        deltas = np.concatenate([[0], np.diff(abs_indices)])
        
        # 构建序列 - 使用0-based索引
        seq = []
        for idx, (_, row) in enumerate(group_sorted.iterrows()):
            seq.append({
                "d": int(row["d_0based"]),     # 0-based 日期
                "t": int(row["t_0based"]),     # 0-based 时间
                "loc_id": int(row["loc_id"]),  # 0-based 位置ID
                "delta": int(deltas[idx])
            })
        
        user2seq[uid] = seq
        
        # 定期显示进度
        if (i + 1) % 1000 == 0:
            print(f"[INFO] 已处理 {i+1}/{total_users} 个用户")

    print(f"[INFO] 用户序列构建完成")
    return user2seq, grid_width * grid_width

def split_train_test(user2seq, num_test_users=2000):
    """划分训练集和测试集用户"""
    all_uids = sorted(user2seq.keys())
    test_users = all_uids[-num_test_users:]
    train_users = all_uids[:-num_test_users]
    return train_users, test_users

def generate_test_sequences(user2seq, test_users, cache_dir=None):
    """为测试集用户生成输入和标签序列"""
    if cache_dir is None:
        cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_input = os.path.join(cache_dir, "test_user2inputseq.pkl")
    cache_target = os.path.join(cache_dir, "test_user2targetseq.pkl")
    
    if os.path.exists(cache_input) and os.path.exists(cache_target):
        print(f"[INFO] 找到已缓存的测试集数据，直接加载")
        with open(cache_input, "rb") as f:
            test_user2inputseq = pickle.load(f)
        with open(cache_target, "rb") as f:
            test_user2targetseq = pickle.load(f)
        return test_user2inputseq, test_user2targetseq

    print("[INFO] 未找到缓存，开始为测试集用户生成输入/标签序列...")
    test_user2inputseq = {}
    test_user2targetseq = {}
    
    for uid in tqdm(test_users, desc="生成测试集序列"):
        seq = user2seq[uid]
        input_seq = []
        target_seq = []
        
        # 修正：基于0-based索引计算cutoff_date
        # 原始数据 d 是1-75，转换后是0-74
        # 最后15天应该是60-74 (0-based)，所以cutoff是60
        all_dates = [item["d"] for item in seq]
        max_date_in_seq = max(all_dates)
        cutoff_date = max_date_in_seq - 15 + 1  # 如果max_date是74，cutoff=60
        
        for item in seq:
            d, t, loc_id, delta = item["d"], item["t"], item["loc_id"], item["delta"]
            if d < cutoff_date:
                input_seq.append({"d": d, "t": t, "loc_id": loc_id, "delta": delta})
            else:
                target_seq.append({"d": d, "t": t, "loc_id": loc_id, "delta": delta})
        
        test_user2inputseq[uid] = input_seq
        test_user2targetseq[uid] = target_seq

    # 缓存到磁盘
    with open(cache_input, "wb") as f:
        pickle.dump(test_user2inputseq, f)
    with open(cache_target, "wb") as f:
        pickle.dump(test_user2targetseq, f)
    print(f"[INFO] 已缓存测试集数据")
    return test_user2inputseq, test_user2targetseq

# ------------------------------
# 6. PyTorch Dataset 定义
# ------------------------------

class LPBertTrainDataset(Dataset):
    """训练集 Dataset - 存储原始序列，mask在collate中动态生成"""
    def __init__(self, user2seq, train_users):
        super().__init__()
        self.train_users = sorted(train_users)
        self.user2seq = user2seq

    def __len__(self):
        return len(self.train_users)

    def __getitem__(self, idx):
        uid = self.train_users[idx]
        seq = self.user2seq[uid]
        return seq

class LPBertTestDataset(Dataset):
    """测试集 Dataset"""
    def __init__(self, test_user2inputseq, test_user2targetseq):
        super().__init__()
        self.uids = sorted(test_user2inputseq.keys())
        self.user2input = test_user2inputseq
        self.user2target = test_user2targetseq

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        input_seq = self.user2input[uid]
        target_seq = self.user2target[uid]
        return input_seq, target_seq

# ------------------------------
# 7. Collate 函数（支持在线Mask）
# ------------------------------

def collate_fn_train_online_mask(batch_list, num_locations, max_timedelta=None, mask_days=15):
    """训练集 collate 函数 - 支持在线随机mask"""
    batch_size = len(batch_list)
    
    # 第一步：对每个序列进行在线mask
    masked_batch_list = []
    
    for original_seq in batch_list:
        if len(original_seq) == 0:
            masked_batch_list.append([])
            continue
        
        # 获取该用户所有出现的日期
        all_dates = sorted(list(set(item["d"] for item in original_seq)))
        
        # 随机选择mask的起始日期
        if len(all_dates) == 0:
            masked_batch_list.append([])
            continue
        elif len(all_dates) <= mask_days:
            # 如果总天数不足，mask所有日期
            mask_start_date = all_dates[0]
            mask_end_date = all_dates[-1]
        else:
            # 找到可以完整mask 15天的起始位置
            d_min, d_max = all_dates[0], all_dates[-1]
            
            if d_max - d_min + 1 <= mask_days:
                # 日期跨度不足15天，mask整个区间
                mask_start_date = d_min
                mask_end_date = d_max
            else:
                # 随机选择起始日期，确保能覆盖15天
                mask_start_date = random.randint(d_min, d_max - mask_days + 1)
                mask_end_date = mask_start_date + mask_days - 1
        
        # 创建masked序列
        masked_seq = []
        for item in original_seq:
            d = item["d"]
            
            if mask_start_date <= d <= mask_end_date:
                # 在mask范围内
                masked_item = {
                    "d": item["d"],
                    "t": item["t"],
                    "loc_id": None,  # mask掉
                    "delta": item["delta"],
                    "target_loc": item["loc_id"]  # 保存真实位置作为标签
                }
            else:
                # 不在mask范围内
                masked_item = {
                    "d": item["d"],
                    "t": item["t"],
                    "loc_id": item["loc_id"],
                    "delta": item["delta"],
                    "target_loc": -1  # 不需要预测
                }
            
            masked_seq.append(masked_item)
        
        masked_batch_list.append(masked_seq)
    
    # 第二步：原有的collate逻辑
    lengths = [len(seq) for seq in masked_batch_list]
    if not lengths:
        return (torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), 
                torch.empty(0), torch.empty(0), torch.empty(0))
    
    L_max = max(lengths)
    
    if max_timedelta is None:
        max_timedelta = max((item["delta"] for seq in masked_batch_list for item in seq), default=0)

    # 创建张量
    date_tensor = torch.zeros((batch_size, L_max), dtype=torch.long)
    time_tensor = torch.zeros((batch_size, L_max), dtype=torch.long)
    loc_tensor = torch.zeros((batch_size, L_max), dtype=torch.long)
    delta_tensor = torch.zeros((batch_size, L_max), dtype=torch.long)
    padding_mask = torch.zeros((batch_size, L_max), dtype=torch.bool)
    token_mask = torch.zeros((batch_size, L_max), dtype=torch.bool)
    target_locations = torch.full((batch_size, L_max), -1, dtype=torch.long)

    for i, seq in enumerate(masked_batch_list):
        L = len(seq)
        for j, item in enumerate(seq):
            d, t, loc_id, delta = item["d"], item["t"], item["loc_id"], item["delta"]
            
            # 确保数值在合理范围内
            date_tensor[i, j] = max(0, d)  # 确保非负
            time_tensor[i, j] = max(0, t)  # 确保非负
            
            true_loc = item.get("target_loc", -1)
            if true_loc >= 0:
                target_locations[i, j] = min(true_loc, num_locations - 1)  # 确保不超出范围
            else:
                target_locations[i, j] = -1
            
            if loc_id is None:
                token_mask[i, j] = True
                loc_tensor[i, j] = num_locations  # mask token
            else:
                # 确保 loc_id 不超出范围
                loc_tensor[i, j] = min(max(0, loc_id), num_locations - 1)
                
            # 确保 delta 不超出范围
            delta_tensor[i, j] = min(max(0, delta), max_timedelta)
            padding_mask[i, j] = True
            
    return date_tensor, time_tensor, loc_tensor, delta_tensor, padding_mask, token_mask, target_locations

def collate_fn_test(batch_list, num_locations):
    """测试集 collate 函数"""
    batch_size = len(batch_list)
    input_lens = [len(inp) for inp, _ in batch_list]
    target_lens = [len(tgt) for _, tgt in batch_list]
    
    if not input_lens or not target_lens:
        return (torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0),
                torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
    
    L_in_max = max(input_lens) if input_lens else 1
    L_t_max = max(target_lens) if target_lens else 1

    # 输入序列张量
    date_in = torch.zeros((batch_size, L_in_max), dtype=torch.long)
    time_in = torch.zeros((batch_size, L_in_max), dtype=torch.long)
    loc_in = torch.zeros((batch_size, L_in_max), dtype=torch.long)
    delta_in = torch.zeros((batch_size, L_in_max), dtype=torch.long)
    padding_in = torch.zeros((batch_size, L_in_max), dtype=torch.bool)

    # 目标序列张量
    date_t = torch.zeros((batch_size, L_t_max), dtype=torch.long)
    time_t = torch.zeros((batch_size, L_t_max), dtype=torch.long)
    loc_t = torch.zeros((batch_size, L_t_max), dtype=torch.long)
    delta_t = torch.zeros((batch_size, L_t_max), dtype=torch.long)
    padding_t = torch.zeros((batch_size, L_t_max), dtype=torch.bool)

    for i, (inp, tgt) in enumerate(batch_list):
        # 处理输入序列
        for j, item in enumerate(inp):
            date_in[i, j] = item["d"]
            time_in[i, j] = item["t"]
            loc_in[i, j] = item["loc_id"]
            delta_in[i, j] = item["delta"]
            padding_in[i, j] = True
        
        # 处理目标序列
        for j, item in enumerate(tgt):
            date_t[i, j] = item["d"]
            time_t[i, j] = item["t"]
            loc_t[i, j] = item["loc_id"]
            delta_t[i, j] = item["delta"]
            padding_t[i, j] = True

    return date_in, time_in, loc_in, delta_in, padding_in, date_t, time_t, loc_t, delta_t, padding_t

# ------------------------------
# 8. LP-BERT 模型定义
# ------------------------------

class LPBertModel(nn.Module):
    def __init__(self, num_dates, num_times, num_locations, max_timedelta, emb_dim=128,
                 max_pos_emb=4096, num_transformer_layers=4, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_locations = num_locations
        self.mask_token_id = num_locations

        print(f"[INFO] 创建模型 embedding 层 (0-based索引):")
        print(f"  - num_dates: {num_dates} (支持 0-{num_dates-1})")
        print(f"  - num_times: {num_times} (支持 0-{num_times-1})")
        print(f"  - num_locations: {num_locations} (支持 0-{num_locations-1}, mask_token: {self.mask_token_id})")
        print(f"  - max_timedelta: {max_timedelta} (支持 0-{max_timedelta})")
        print(f"[INFO] 模型架构参数:")
        print(f"  - emb_dim: {emb_dim}")
        print(f"  - num_transformer_layers: {num_transformer_layers}")
        print(f"  - nhead: {nhead}")
        print(f"  - dim_feedforward: {dim_feedforward}")
        print(f"  - dropout: {dropout}")

        # Embedding 层 - 现在数据已经是0-based，不需要额外的安全边界
        self.date_emb = nn.Embedding(num_dates, emb_dim)
        self.time_emb = nn.Embedding(num_times, emb_dim)
        self.loc_emb = nn.Embedding(num_locations + 1, emb_dim)  # +1 for mask token
        self.delta_emb = nn.Embedding(max_timedelta + 1, emb_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # 分类头
        self.classifier = nn.Linear(emb_dim, num_locations)
        
        # 参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INFO] 模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    def forward(self, date_idx, time_idx, loc_idx, delta_idx, padding_mask, token_mask):
        if date_idx.numel() == 0:
            return torch.empty((0, self.num_locations), device=date_idx.device)
        
        # 确保数据在正确范围内 (现在应该不需要，但保持作为安全检查)
        date_idx = torch.clamp(date_idx, 0, self.date_emb.num_embeddings - 1)
        time_idx = torch.clamp(time_idx, 0, self.time_emb.num_embeddings - 1)
        loc_idx = torch.clamp(loc_idx, 0, self.loc_emb.num_embeddings - 1)
        delta_idx = torch.clamp(delta_idx, 0, self.delta_emb.num_embeddings - 1)
        
        # Embedding
        date_e = self.date_emb(date_idx)
        time_e = self.time_emb(time_idx)
        loc_e = self.loc_emb(loc_idx)
        delta_e = self.delta_emb(delta_idx)

        # 特征融合
        x = date_e + time_e + loc_e + delta_e

        # Transformer
        src_key_padding_mask = ~padding_mask
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 分类
        logits = self.classifier(out)

        # 提取被 mask 位置的 logits
        masked_indices = token_mask.nonzero(as_tuple=False)
        if masked_indices.size(0) == 0:
            return torch.empty((0, self.num_locations), device=x.device)

        batch_idx = masked_indices[:, 0]
        seq_idx = masked_indices[:, 1]
        logits_masked = logits[batch_idx, seq_idx, :]

        return logits_masked

def compute_mask_loss(logits_masked, target_locations, token_mask):
    """计算 mask 位置的损失"""
    device = logits_masked.device
    
    masked_indices = token_mask.nonzero(as_tuple=False)
    if masked_indices.size(0) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    batch_idx = masked_indices[:, 0]
    seq_idx = masked_indices[:, 1]
    
    target_valid = target_locations[batch_idx, seq_idx]
    valid_mask = target_valid >= 0
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    logits_valid = logits_masked[valid_mask]
    target_valid = target_valid[valid_mask]
    
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits_valid, target_valid)
    return loss

# ------------------------------
# 9. 学习率调度器
# ------------------------------

def get_linear_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """创建带warmup的线性学习率调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ------------------------------
# 10. 主函数
# ------------------------------

def main():
    args = parse_args()

    # 获取分布式训练参数
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    torch.cuda.set_device(local_rank)
    
    # 检查缓存状态
    base_cache_path = os.path.join(args.cache_dir, "base_data.pkl")
    cache_exists = os.path.exists(base_cache_path)
    
    # 缓存检查和提示
    if not cache_exists:
        if world_size > 1:
            print(f"[INFO] 检测到多GPU训练但基础缓存不存在。请先运行单GPU预处理:")
            print(f"python {sys.argv[0]} --raw_csv_path {args.raw_csv_path} --cache_dir {args.cache_dir} --batch_size 4 --epochs 1")
            print(f"[INFO] 预处理完成后再运行多GPU训练")
            sys.exit(1)
        else:
            print("[INFO] 单GPU模式，开始数据预处理...")
    else:
        print("[INFO] 发现基础数据缓存，跳过原始数据预处理步骤")
    
    # 初始化分布式训练
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
    
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[INFO] 训练模式: {'多GPU DDP' if world_size > 1 else '单GPU'}")
        print(f"[INFO] world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print(f"[INFO] 命令行参数:")
        for arg, value in vars(args).items():
            print(f"  --{arg}: {value}")

    # 数据预处理（仅在基础缓存不存在时执行）
    if not cache_exists:
        print("[INFO] 开始基础数据预处理...")
        try:
            # 读取和处理原始数据
            raw_df = load_raw_data(args.raw_csv_path)
            user2seq_all, num_locations = build_user_sequences(raw_df)
            del raw_df  # 清理内存
            
            # 计算数据统计信息 - 基于0-based索引
            max_date = max(item["d"] for seq in user2seq_all.values() for item in seq) + 1  # 0-74 -> 75
            max_time = max(item["t"] for seq in user2seq_all.values() for item in seq) + 1  # 0-47 -> 48
            max_delta = max(item["delta"] for seq in user2seq_all.values() for item in seq)

            print(f"[INFO] 用户总数：{len(user2seq_all)}, 地点种类：{num_locations}")
            print(f"[INFO] max_date={max_date}, max_time={max_time}, max_delta={max_delta}")
            print(f"[INFO] 数据使用0-based索引: d∈[0,{max_date-1}], t∈[0,{max_time-1}], loc_id∈[0,{num_locations-1}]")

            # 划分训练测试集
            train_users, test_users = split_train_test(user2seq_all, num_test_users=2000)
            print(f"[INFO] 训练用户：{len(train_users)}, 测试用户：{len(test_users)}")

            # 生成测试集（测试集是固定的，可以缓存）
            test_cache_dir = os.path.join(args.cache_dir, "test")
            test_user2input, test_user2target = generate_test_sequences(
                user2seq_all, test_users, cache_dir=test_cache_dir
            )

            # 保存基础数据（不包含训练集的具体mask）
            base_data = {
                'user2seq_all': user2seq_all,
                'num_locations': num_locations,
                'max_date': max_date,
                'max_time': max_time,
                'max_delta': max_delta,
                'train_users': train_users,
                'test_users': test_users,
                'test_user2input': test_user2input,
                'test_user2target': test_user2target
            }
            
            os.makedirs(args.cache_dir, exist_ok=True)
            with open(base_cache_path, "wb") as f:
                pickle.dump(base_data, f)
            
            print(f"[INFO] 基础数据预处理完成，缓存保存到: {args.cache_dir}")
            
        except Exception as e:
            print(f"[ERROR] 数据预处理失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    # 加载基础数据
    print("[INFO] 加载基础数据...")
    with open(base_cache_path, "rb") as f:
        base_data = pickle.load(f)
    
    user2seq_all = base_data['user2seq_all']
    num_locations = base_data['num_locations']
    max_date = base_data['max_date']
    max_time = base_data['max_time']
    max_delta = base_data['max_delta']
    train_users = base_data['train_users']
    test_users = base_data['test_users']
    test_user2input = base_data['test_user2input']
    test_user2target = base_data['test_user2target']

    if rank == 0:
        print(f"[INFO] 数据加载完成")
        print(f"[INFO] 训练数据集: {len(train_users)} 用户")
        print(f"[INFO] 测试数据集: {len(test_user2input)} 用户")

    # 创建数据加载器
    batch_size_per_gpu = max(1, args.batch_size // world_size)
    if rank == 0 and args.batch_size % world_size != 0:
        print(f"[WARNING] batch_size {args.batch_size} 不能被 world_size {world_size} 整除")
        print(f"[INFO] 每个GPU实际batch_size: {batch_size_per_gpu}")

    # 训练集 - 使用新的Dataset，不预先生成mask
    train_dataset = LPBertTrainDataset(user2seq_all, train_users)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=lambda batch: collate_fn_train_online_mask(batch, num_locations, max_delta, mask_days=args.mask_days),
        num_workers=4,
        pin_memory=True,
    )

    # 测试集
    test_dataset = LPBertTestDataset(test_user2input, test_user2target)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_per_gpu,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_test(batch, num_locations),
        num_workers=2,
        pin_memory=True,
    )

    if rank == 0:
        print("[INFO] DataLoader 准备完毕")
        print(f"[INFO] 使用在线Mask策略，每个batch动态生成mask")
        print(f"[INFO] 每个epoch的batch数: {len(train_loader)}")

    # 创建模型
    model = LPBertModel(
        num_dates=max_date,
        num_times=max_time,
        num_locations=num_locations,
        max_timedelta=max_delta,
        emb_dim=args.emb_dim,
        max_pos_emb=args.max_pos_emb,
        num_transformer_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 创建学习率调度器
    num_training_steps = args.epochs * len(train_loader)
    warmup_scheduler = None
    
    if args.warmup_steps > 0:
        warmup_scheduler = get_linear_warmup_scheduler(
            optimizer, 
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )
        if rank == 0:
            print(f"[INFO] 使用Warmup学习率调度，warmup步数: {args.warmup_steps}")
    
    # ReduceLROnPlateau调度器
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        threshold=args.scheduler_threshold,
        threshold_mode='rel',
        verbose=True if rank == 0 else False,
        min_lr=1e-6
    )
    
    # 混合精度训练
    scaler = GradScaler() if args.use_amp else None
    
    # 检查是否需要加载checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint_path, loaded_epoch = find_latest_checkpoint(args.cache_dir)
        if checkpoint_path and os.path.exists(checkpoint_path):
            if rank == 0:
                print(f"[INFO] 找到checkpoint: {checkpoint_path}")
                print(f"[INFO] 正在加载checkpoint...")
            
            try:
                start_epoch, last_loss = load_checkpoint(
                    model, optimizer, plateau_scheduler, scaler, checkpoint_path, device
                )
                global_step = start_epoch * len(train_loader)
                if rank == 0:
                    print(f"[INFO] 成功加载checkpoint，从epoch {start_epoch + 1} 继续训练")
                    print(f"[INFO] 上次训练的loss: {last_loss:.4f}")
                    print(f"[INFO] 当前学习率: {optimizer.param_groups[0]['lr']:.2e}")
            except Exception as e:
                if rank == 0:
                    print(f"[WARNING] 加载checkpoint失败: {e}")
                    print("[INFO] 将从头开始训练")
                start_epoch = 0
                global_step = 0
        else:
            if rank == 0:
                print("[INFO] 未找到checkpoint，从头开始训练")

    # DDP 包装（必须在加载checkpoint之后）
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if rank == 0:
        print("[INFO] 模型初始化完毕，开始训练...")
        print(f"[INFO] 训练将从epoch {start_epoch + 1} 开始，目标epoch {args.epochs}")
        print(f"[INFO] 初始学习率: {args.lr:.2e}")
        print(f"[INFO] Batch size: {args.batch_size} (每GPU: {batch_size_per_gpu})")
        print(f"[INFO] 使用混合精度训练: {'是' if args.use_amp else '否'}")

    # 用于存储评估指标的历史
    metrics_history = []

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        batch_geo_bleu_sum = 0.0
        num_eval_batches = 0

        if rank == 0:
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            batch_iter = train_loader
            
        for batch_idx, batch_data in enumerate(batch_iter):
            global_step += 1
            
            # 数据移动到GPU
            date_tensor, time_tensor, loc_tensor, delta_tensor, padding_mask, token_mask, target_locations = batch_data
            date_tensor = date_tensor.to(device, non_blocking=True)
            time_tensor = time_tensor.to(device, non_blocking=True)
            loc_tensor = loc_tensor.to(device, non_blocking=True)
            delta_tensor = delta_tensor.to(device, non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True)
            token_mask = token_mask.to(device, non_blocking=True)
            target_locations = target_locations.to(device, non_blocking=True)

            try:
                # 混合精度训练
                if args.use_amp:
                    with autocast():
                        logits_masked = model(date_tensor, time_tensor, loc_tensor, delta_tensor, padding_mask, token_mask)
                        loss = compute_mask_loss(logits_masked, target_locations, token_mask)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 前向传播
                    logits_masked = model(date_tensor, time_tensor, loc_tensor, delta_tensor, padding_mask, token_mask)
                    loss = compute_mask_loss(logits_masked, target_locations, token_mask)
                    
                    # 检查损失有效性
                    if torch.isnan(loss) or torch.isinf(loss):
                        if rank == 0:
                            print(f"[WARNING] 无效损失值: {loss.item()}, 跳过此batch")
                        continue
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Warmup学习率调整
                if warmup_scheduler is not None and global_step < args.warmup_steps:
                    warmup_scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1

                # 周期性评估Geo-BLEU
                if batch_idx > 0 and batch_idx % args.eval_interval == 0:
                    # 将整个batch数据传递给评估函数
                    batch_data_gpu = (date_tensor, time_tensor, loc_tensor, delta_tensor, 
                                     padding_mask, token_mask, target_locations)
                    
                    # 如果是DDP模型，使用module属性
                    model_for_eval = model.module if hasattr(model, 'module') else model
                    
                    geo_bleu = evaluate_batch_geo_bleu(
                        model_for_eval, 
                        batch_data_gpu, 
                        device, 
                        num_locations, 
                        args.geo_bleu_n
                    )
                    
                    batch_geo_bleu_sum += geo_bleu
                    num_eval_batches += 1
                    
                    if rank == 0:
                        avg_geo_bleu = batch_geo_bleu_sum / num_eval_batches
                        current_lr = optimizer.param_groups[0]['lr']
                        batch_iter.set_postfix(
                            loss=f"{loss.item():.4f}", 
                            geo_bleu=f"{geo_bleu:.4f}",
                            avg_geo_bleu=f"{avg_geo_bleu:.4f}",
                            lr=f"{current_lr:.2e}"
                        )
                        
                        # 记录指标
                        metrics_history.append({
                            'epoch': epoch + 1,
                            'batch': batch_idx,
                            'loss': loss.item(),
                            'geo_bleu': geo_bleu,
                            'lr': current_lr
                        })
                else:
                    if rank == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        batch_iter.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
                    
            except Exception as e:
                if rank == 0:
                    print(f"[ERROR] Batch {batch_idx} 失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue

        # Epoch 结束
        if rank == 0:
            batch_iter.close()

        avg_loss = epoch_loss / max(1, num_batches)
        avg_geo_bleu = batch_geo_bleu_sum / max(1, num_eval_batches) if num_eval_batches > 0 else 0.0
        
        # 调整学习率（只在warmup结束后使用plateau scheduler）
        if global_step >= args.warmup_steps:
            plateau_scheduler.step(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if rank == 0:
            print(f"[INFO] Epoch {epoch+1} 完成")
            print(f"  - 平均损失: {avg_loss:.4f}")
            print(f"  - 平均Geo-BLEU: {avg_geo_bleu:.4f}")
            print(f"  - 当前学习率: {current_lr:.2e}")
            
            # 记录训练日志
            log_training_info(args.cache_dir, epoch+1, avg_loss, avg_geo_bleu, current_lr)
            
            # 保存模型
            model_to_save = model.module if hasattr(model, 'module') else model
            ckpt_path = os.path.join(args.cache_dir, f"checkpoint_epoch{epoch+1}.pt")
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': plateau_scheduler.state_dict(),
                'loss': avg_loss,
                'geo_bleu': avg_geo_bleu,
                'metrics_history': metrics_history,
                'model_config': {
                    'num_dates': max_date,
                    'num_times': max_time,
                    'num_locations': num_locations,
                    'max_timedelta': max_delta,
                    'emb_dim': args.emb_dim,
                    'max_pos_emb': args.max_pos_emb,
                    'num_transformer_layers': args.num_layers,
                    'nhead': args.nhead,
                    'dim_feedforward': args.dim_feedforward,
                    'dropout': args.dropout
                },
                'args': vars(args)
            }
            
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(save_dict, ckpt_path)
            print(f"[INFO] 模型已保存: {ckpt_path}")
            
            # 保存指标历史
            metrics_path = os.path.join(args.cache_dir, "metrics_history.pkl")
            with open(metrics_path, "wb") as f:
                pickle.dump(metrics_history, f)

    # 清理
    if world_size > 1:
        dist.destroy_process_group()
    if rank == 0:
        print("[INFO] 训练完成!")
        print(f"[INFO] 最终平均Geo-BLEU: {avg_geo_bleu:.4f}")
        
        # 打印训练总结
        summary_path = os.path.join(args.cache_dir, "training_summary.txt")
        with open(summary_path, "w") as f:
            f.write("=== LP-BERT 训练总结 ===\n")
            f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总epoch数: {args.epochs}\n")
            f.write(f"最终损失: {avg_loss:.6f}\n")
            f.write(f"最终Geo-BLEU: {avg_geo_bleu:.6f}\n")
            f.write(f"最终学习率: {current_lr:.2e}\n")
            f.write("\n模型配置:\n")
            f.write(f"  - Embedding维度: {args.emb_dim}\n")
            f.write(f"  - Transformer层数: {args.num_layers}\n")
            f.write(f"  - 注意力头数: {args.nhead}\n")
            f.write(f"  - 前馈层维度: {args.dim_feedforward}\n")
            f.write(f"  - Batch size: {args.batch_size}\n")
            f.write(f"  - 初始学习率: {args.lr}\n")
            f.write(f"  - Warmup步数: {args.warmup_steps}\n")
            f.write(f"  - 使用混合精度: {'是' if args.use_amp else '否'}\n")
        
        print(f"[INFO] 训练总结已保存到: {summary_path}")

if __name__ == "__main__":
    main()