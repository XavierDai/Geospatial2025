#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lpbert_xy_ddp.py

改进的 LP-BERT 分布式训练脚本，整合了关键改进：
- 分离 X/Y 预测（核心改进）
- 添加验证集（固定mask 60-74天）
- 统一数据格式（1-based索引，201作为mask token）
- 改进的训练策略

使用方法：
1. 先运行单GPU预处理：
   python lpbert_model_xy_isolated.py --raw_csv_path ./yjmob100k-dataset1.csv --cache_dir ./cache_lpbert_xy --batch_size 4 --epochs 1

2. 再运行多GPU训练：
   torchrun --nproc_per_node=4 --master_port=12345 lpbert_model_xy_isolated.py \
     --raw_csv_path ./yjmob100k-dataset1.csv \
     --cache_dir ./cache_lpbert_xy \
     --batch_size 32 \
     --epochs 200 \
     --lr 2e-5
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
    parser = argparse.ArgumentParser(description="LP-BERT XY Distributed Training Script")

    # 数据相关参数
    parser.add_argument(
        "--raw_csv_path", type=str, default="./yjmob100k-dataset1.csv",
        help="原始 CSV 数据路径"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./cache_lpbert_xy",
        help="预处理后数据的缓存目录"
    )
    parser.add_argument(
        "--mask_days", type=int, default=15,
        help="对训练集中每个用户随机连续 Mask 的天数"
    )
    parser.add_argument(
        "--val_mask_prob", type=float, default=0.3,
        help="训练时使用验证集mask策略（60-74天）的概率"
    )
    
    # 训练相关参数
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="全局 Batch Size（DDP 会分摊到每张卡上）"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="训练轮数"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
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
    
    # 评估相关参数
    parser.add_argument(
        "--eval_interval", type=int, default=100,
        help="每多少个batch计算一次Geo-BLEU"
    )
    parser.add_argument(
        "--val_interval", type=int, default=500,
        help="每多少个batch在验证集上评估"
    )
    parser.add_argument(
        "--geo_bleu_n", type=int, default=4,
        help="Geo-BLEU的最大n-gram"
    )
    
    # 学习率调度器参数
    parser.add_argument(
        "--scheduler_type", type=str, default="cosine", choices=["cosine", "plateau"],
        help="学习率调度器类型"
    )

    return parser.parse_args()

# ------------------------------
# 2. Geo-BLEU 计算（针对x,y分离）
# ------------------------------

def extract_ngrams(sequence, n):
    """提取序列的n-gram"""
    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i+n])
        ngrams.append(ngram)
    return ngrams

def compute_geo_bleu_xy(pred_x_sequences, pred_y_sequences, target_x_sequences, target_y_sequences, max_n=4):
    """
    计算基于(x,y)坐标对的Geo-BLEU分数
    """
    if len(pred_x_sequences) == 0 or len(target_x_sequences) == 0:
        return 0.0
    
    bleu_scores = []
    
    for pred_x, pred_y, target_x, target_y in zip(pred_x_sequences, pred_y_sequences, 
                                                   target_x_sequences, target_y_sequences):
        # 将x,y组合成坐标对
        pred_seq = [(x, y) for x, y in zip(pred_x, pred_y)]
        target_seq = [(x, y) for x, y in zip(target_x, target_y)]
        
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
        
        bleu_scores.append(geo_mean)
    
    return np.mean(bleu_scores) if bleu_scores else 0.0

# ------------------------------
# 3. 数据预处理（改进版）
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

def build_user_sequences_xy(df):
    """构建用户序列，保持x和y分离"""
    print(f"[INFO] 开始构建用户序列（X/Y分离版本）...")
    
    # 检查数据范围
    print(f"[INFO] 原始数据范围:")
    print(f"  uid: {df['uid'].min()} - {df['uid'].max()}")
    print(f"  d: {df['d'].min()} - {df['d'].max()}")  
    print(f"  t: {df['t'].min()} - {df['t'].max()}")
    print(f"  x: {df['x'].min()} - {df['x'].max()}")
    print(f"  y: {df['y'].min()} - {df['y'].max()}")
    
    # 按用户分组
    grouped = df.groupby('uid')
    user2seq = {}
    
    print(f"[INFO] 处理 {len(grouped)} 个用户的序列...")
    
    for uid, group in tqdm(grouped, desc="构建用户序列"):
        # 计算绝对时间索引用于排序
        group = group.copy()
        group['abs_idx'] = (group['d'] - 1) * 48 + group['t']
        group_sorted = group.sort_values('abs_idx')
        
        # 计算时间间隔
        abs_indices = group_sorted['abs_idx'].values
        deltas = np.concatenate([[0], np.diff(abs_indices)])
        deltas = np.clip(deltas, 0, 47)  # 限制最大值为47
        
        # 构建序列 - 保持1-based索引，与参考代码一致
        seq = []
        for idx, (_, row) in enumerate(group_sorted.iterrows()):
            seq.append({
                "d": int(row["d"]),      # 1-75
                "t": int(row["t"]) + 1,  # 0-47 -> 1-48
                "x": int(row["x"]),      # 1-200
                "y": int(row["y"]),      # 1-200
                "delta": int(deltas[idx])
            })
        
        user2seq[uid] = seq
    
    print(f"[INFO] 用户序列构建完成")
    return user2seq

def split_train_val_test(user2seq, num_test_users=2000, val_ratio=0.2):
    """划分训练集、验证集和测试集"""
    all_uids = sorted(user2seq.keys())
    
    # 最后2000个用户作为测试集
    test_users = all_uids[-num_test_users:]
    remaining_users = all_uids[:-num_test_users]
    
    # 剩余用户的20%作为验证集
    num_val_users = int(len(remaining_users) * val_ratio)
    val_users = remaining_users[-num_val_users:]
    train_users = remaining_users[:-num_val_users]
    
    return train_users, val_users, test_users

def generate_masked_sequences(user2seq, users, mask_start=60, mask_end=74, cache_dir=None, prefix=""):
    """为用户生成固定mask的序列（用于验证集和测试集）"""
    if cache_dir is None:
        cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_input = os.path.join(cache_dir, f"{prefix}_user2inputseq.pkl")
    cache_target = os.path.join(cache_dir, f"{prefix}_user2targetseq.pkl")
    
    if os.path.exists(cache_input) and os.path.exists(cache_target):
        print(f"[INFO] 找到已缓存的{prefix}数据，直接加载")
        with open(cache_input, "rb") as f:
            user2inputseq = pickle.load(f)
        with open(cache_target, "rb") as f:
            user2targetseq = pickle.load(f)
        return user2inputseq, user2targetseq

    print(f"[INFO] 生成{prefix}数据（mask天数: {mask_start}-{mask_end}）...")
    user2inputseq = {}
    user2targetseq = {}
    
    for uid in tqdm(users, desc=f"生成{prefix}序列"):
        seq = user2seq[uid]
        input_seq = []
        target_seq = []
        
        for item in seq:
            d = item["d"]
            if d < mask_start or d > mask_end:
                # 不在mask范围内，作为输入
                input_seq.append(item)
            else:
                # 在mask范围内，作为目标
                target_seq.append(item)
        
        user2inputseq[uid] = input_seq
        user2targetseq[uid] = target_seq

    # 缓存
    with open(cache_input, "wb") as f:
        pickle.dump(user2inputseq, f)
    with open(cache_target, "wb") as f:
        pickle.dump(user2targetseq, f)
    
    print(f"[INFO] {prefix}数据生成完成")
    return user2inputseq, user2targetseq

# ------------------------------
# 4. Dataset 定义（改进版）
# ------------------------------

class LPBertTrainDatasetXY(Dataset):
    """训练集 Dataset - X/Y分离版本"""
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

class LPBertValTestDatasetXY(Dataset):
    """验证/测试集 Dataset - X/Y分离版本"""
    def __init__(self, user2inputseq, user2targetseq):
        super().__init__()
        self.uids = sorted(user2inputseq.keys())
        self.user2input = user2inputseq
        self.user2target = user2targetseq

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        input_seq = self.user2input[uid]
        target_seq = self.user2target[uid]
        return input_seq, target_seq

# ------------------------------
# 5. Collate 函数（X/Y分离版本）
# ------------------------------

def collate_fn_train_xy(batch_list, mask_days=15, val_mask_prob=0.3):
    """训练集 collate 函数 - X/Y分离版本"""
    batch_size = len(batch_list)
    
    # 对每个序列进行mask
    masked_batch_list = []
    
    for original_seq in batch_list:
        if len(original_seq) == 0:
            masked_batch_list.append([])
            continue
        
        # 获取该用户所有出现的日期
        all_dates = sorted(list(set(item["d"] for item in original_seq)))
        
        # 决定是否使用验证集的mask策略
        use_val_mask = random.random() < val_mask_prob
        
        if use_val_mask and max(all_dates) >= 74:
            # 使用验证集策略：mask 60-74天
            mask_start_date = 60
            mask_end_date = 74
        else:
            # 随机mask策略
            if len(all_dates) <= mask_days:
                mask_start_date = all_dates[0]
                mask_end_date = all_dates[-1]
            else:
                d_min, d_max = all_dates[0], all_dates[-1]
                if d_max - d_min + 1 <= mask_days:
                    mask_start_date = d_min
                    mask_end_date = d_max
                else:
                    mask_start_date = random.randint(d_min, max(d_min, d_max - mask_days + 1))
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
                    "x": 201,  # mask token
                    "y": 201,  # mask token
                    "delta": item["delta"],
                    "target_x": item["x"] - 1,  # 转换为0-199
                    "target_y": item["y"] - 1   # 转换为0-199
                }
            else:
                # 不在mask范围内
                masked_item = {
                    "d": item["d"],
                    "t": item["t"],
                    "x": item["x"],
                    "y": item["y"],
                    "delta": item["delta"],
                    "target_x": -1,  # 不需要预测
                    "target_y": -1   # 不需要预测
                }
            
            masked_seq.append(masked_item)
        
        masked_batch_list.append(masked_seq)
    
    # 找到最大长度
    lengths = [len(seq) for seq in masked_batch_list]
    if not lengths:
        return None
    
    L_max = max(lengths)
    
    # 创建张量 - 使用0作为padding
    date_tensor = torch.zeros((batch_size, L_max), dtype=torch.long)
    time_tensor = torch.zeros((batch_size, L_max), dtype=torch.long)
    x_tensor = torch.zeros((batch_size, L_max), dtype=torch.long)
    y_tensor = torch.zeros((batch_size, L_max), dtype=torch.long)
    delta_tensor = torch.zeros((batch_size, L_max), dtype=torch.long)
    padding_mask = torch.zeros((batch_size, L_max), dtype=torch.bool)
    pred_mask = torch.zeros((batch_size, L_max), dtype=torch.bool)
    target_x = torch.full((batch_size, L_max), -1, dtype=torch.long)
    target_y = torch.full((batch_size, L_max), -1, dtype=torch.long)
    
    for i, seq in enumerate(masked_batch_list):
        L = len(seq)
        for j, item in enumerate(seq):
            date_tensor[i, j] = item["d"]
            time_tensor[i, j] = item["t"]
            x_tensor[i, j] = item["x"]
            y_tensor[i, j] = item["y"]
            delta_tensor[i, j] = item["delta"]
            padding_mask[i, j] = True
            
            if item["x"] == 201:  # mask token
                pred_mask[i, j] = True
                target_x[i, j] = item["target_x"]
                target_y[i, j] = item["target_y"]
    
    return {
        'd': date_tensor,
        't': time_tensor,
        'x': x_tensor,
        'y': y_tensor,
        'delta': delta_tensor,
        'padding_mask': padding_mask,
        'pred_mask': pred_mask,
        'target_x': target_x,
        'target_y': target_y,
        'lengths': torch.tensor(lengths)
    }

def collate_fn_val_test_xy(batch_list):
    """验证/测试集 collate 函数 - X/Y分离版本"""
    batch_size = len(batch_list)
    
    # 分离输入和目标
    input_seqs = []
    target_seqs = []
    
    for inp, tgt in batch_list:
        # 处理输入序列
        input_seq = []
        for item in inp:
            input_seq.append({
                "d": item["d"],
                "t": item["t"],
                "x": item["x"],
                "y": item["y"],
                "delta": item["delta"]
            })
        
        # 处理目标序列（需要mask）
        target_seq = []
        for item in tgt:
            target_seq.append({
                "d": item["d"],
                "t": item["t"],
                "x": 201,  # mask
                "y": 201,  # mask
                "delta": item["delta"],
                "target_x": item["x"] - 1,  # 转换为0-199
                "target_y": item["y"] - 1   # 转换为0-199
            })
        
        # 合并输入和目标
        full_seq = input_seq + target_seq
        input_seqs.append(full_seq)
        target_seqs.append(target_seq)
    
    # 使用训练集的collate逻辑处理
    lengths = [len(seq) for seq in input_seqs]
    if not lengths:
        return None
    
    L_max = max(lengths)
    
    # 创建张量
    date_tensor = torch.zeros((batch_size, L_max), dtype=torch.long)
    time_tensor = torch.zeros((batch_size, L_max), dtype=torch.long)
    x_tensor = torch.zeros((batch_size, L_max), dtype=torch.long)
    y_tensor = torch.zeros((batch_size, L_max), dtype=torch.long)
    delta_tensor = torch.zeros((batch_size, L_max), dtype=torch.long)
    padding_mask = torch.zeros((batch_size, L_max), dtype=torch.bool)
    pred_mask = torch.zeros((batch_size, L_max), dtype=torch.bool)
    target_x = torch.full((batch_size, L_max), -1, dtype=torch.long)
    target_y = torch.full((batch_size, L_max), -1, dtype=torch.long)
    
    for i, seq in enumerate(input_seqs):
        L = len(seq)
        for j, item in enumerate(seq):
            date_tensor[i, j] = item["d"]
            time_tensor[i, j] = item["t"]
            x_tensor[i, j] = item["x"]
            y_tensor[i, j] = item["y"]
            delta_tensor[i, j] = item["delta"]
            padding_mask[i, j] = True
            
            if item["x"] == 201:  # mask token
                pred_mask[i, j] = True
                target_x[i, j] = item["target_x"]
                target_y[i, j] = item["target_y"]
    
    return {
        'd': date_tensor,
        't': time_tensor,
        'x': x_tensor,
        'y': y_tensor,
        'delta': delta_tensor,
        'padding_mask': padding_mask,
        'pred_mask': pred_mask,
        'target_x': target_x,
        'target_y': target_y,
        'lengths': torch.tensor(lengths)
    }

# ------------------------------
# 6. 模型定义（X/Y分离版本）
# ------------------------------

class LPBertModelXY(nn.Module):
    """LP-BERT模型 - X/Y分离预测版本"""
    def __init__(self, emb_dim=128, num_layers=4, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        self.emb_dim = emb_dim
        
        # Embedding层 - 与参考代码保持一致
        self.date_emb = nn.Embedding(76, emb_dim)      # 0:<pad>, 1-75
        self.time_emb = nn.Embedding(49, emb_dim)      # 0:<pad>, 1-48
        self.x_emb = nn.Embedding(202, emb_dim)        # 0:<pad>, 1-200, 201:<mask>
        self.y_emb = nn.Embedding(202, emb_dim)        # 0:<pad>, 1-200, 201:<mask>
        self.delta_emb = nn.Embedding(48, emb_dim)     # 0-47
        
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
            num_layers=num_layers
        )
        
        # 分离的预测头 - 使用瓶颈结构
        self.ffn_x = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 200)
        )
        self.ffn_y = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 200)
        )
        
        # 参数统计
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[INFO] 模型参数量: {total_params:,}")
    
    def forward(self, date_idx, time_idx, x_idx, y_idx, delta_idx, padding_mask):
        """
        前向传播
        返回: (logits_x, logits_y) - 形状都是 [batch, seq, 200]
        """
        # Embedding
        date_e = self.date_emb(date_idx)
        time_e = self.time_emb(time_idx)
        x_e = self.x_emb(x_idx)
        y_e = self.y_emb(y_idx)
        delta_e = self.delta_emb(delta_idx)
        
        # 特征融合
        x = date_e + time_e + x_e + y_e + delta_e
        
        # Transformer
        src_key_padding_mask = ~padding_mask
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # 分离预测
        logits_x = self.ffn_x(out)  # [batch, seq, 200]
        logits_y = self.ffn_y(out)  # [batch, seq, 200]
        
        return logits_x, logits_y

def compute_loss_xy(logits_x, logits_y, target_x, target_y, pred_mask):
    """计算X和Y的联合损失"""
    device = logits_x.device
    
    # 找到需要预测的位置
    mask_indices = pred_mask.nonzero(as_tuple=False)
    if mask_indices.size(0) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    batch_idx = mask_indices[:, 0]
    seq_idx = mask_indices[:, 1]
    
    # 提取需要预测的logits和targets
    pred_x = logits_x[batch_idx, seq_idx]  # [num_masked, 200]
    pred_y = logits_y[batch_idx, seq_idx]  # [num_masked, 200]
    true_x = target_x[batch_idx, seq_idx]  # [num_masked]
    true_y = target_y[batch_idx, seq_idx]  # [num_masked]
    
    # 过滤有效的目标（>= 0）
    valid_mask = (true_x >= 0) & (true_y >= 0)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    pred_x = pred_x[valid_mask]
    pred_y = pred_y[valid_mask]
    true_x = true_x[valid_mask]
    true_y = true_y[valid_mask]
    
    # 计算损失
    loss_fn = nn.CrossEntropyLoss()
    loss_x = loss_fn(pred_x, true_x)
    loss_y = loss_fn(pred_y, true_y)
    
    return loss_x + loss_y

def evaluate_batch_geo_bleu_xy(model, batch_data, device):
    """评估一个batch的GEO-BLEU（X/Y版本）"""
    model.eval()
    
    with torch.no_grad():
        # 解包数据
        d = batch_data['d'].to(device)
        t = batch_data['t'].to(device)
        x = batch_data['x'].to(device)
        y = batch_data['y'].to(device)
        delta = batch_data['delta'].to(device)
        padding_mask = batch_data['padding_mask'].to(device)
        pred_mask = batch_data['pred_mask'].to(device)
        target_x = batch_data['target_x'].to(device)
        target_y = batch_data['target_y'].to(device)
        
        # 模型预测
        logits_x, logits_y = model(d, t, x, y, delta, padding_mask)
        
        # 获取预测
        pred_x = torch.argmax(logits_x, dim=-1)  # [batch, seq]
        pred_y = torch.argmax(logits_y, dim=-1)  # [batch, seq]
        
        # 提取mask位置的预测和目标
        batch_size = pred_mask.size(0)
        pred_x_seqs = []
        pred_y_seqs = []
        true_x_seqs = []
        true_y_seqs = []
        
        for i in range(batch_size):
            mask_pos = pred_mask[i].nonzero(as_tuple=False).squeeze(-1)
            if len(mask_pos) > 0:
                pred_x_seq = pred_x[i, mask_pos].cpu().tolist()
                pred_y_seq = pred_y[i, mask_pos].cpu().tolist()
                true_x_seq = target_x[i, mask_pos].cpu().tolist()
                true_y_seq = target_y[i, mask_pos].cpu().tolist()
                
                # 过滤有效目标
                valid_indices = [(j, tx, ty) for j, (tx, ty) in enumerate(zip(true_x_seq, true_y_seq)) if tx >= 0 and ty >= 0]
                if valid_indices:
                    indices = [j for j, _, _ in valid_indices]
                    pred_x_seqs.append([pred_x_seq[j] for j in indices])
                    pred_y_seqs.append([pred_y_seq[j] for j in indices])
                    true_x_seqs.append([tx for _, tx, _ in valid_indices])
                    true_y_seqs.append([ty for _, _, ty in valid_indices])
        
        # 计算GEO-BLEU
        geo_bleu = compute_geo_bleu_xy(pred_x_seqs, pred_y_seqs, true_x_seqs, true_y_seqs)
    
    model.train()
    return geo_bleu

# ------------------------------
# 7. Checkpoint 管理
# ------------------------------

def find_latest_checkpoint(cache_dir):
    """在cache_dir中找到最新的checkpoint文件"""
    if not os.path.exists(cache_dir):
        return None, 0
    
    checkpoint_files = []
    pattern = re.compile(r'checkpoint_epoch(\d+)\.pt')
    
    for filename in os.listdir(cache_dir):
        match = pattern.match(filename)
        if match:
            epoch_num = int(match.group(1))
            checkpoint_files.append((filename, epoch_num))
    
    if not checkpoint_files:
        return None, 0
    
    checkpoint_files.sort(key=lambda x: x[1])
    latest_file, latest_epoch = checkpoint_files[-1]
    
    return os.path.join(cache_dir, latest_file), latest_epoch

def save_checkpoint(model, optimizer, scheduler, epoch, loss, geo_bleu, val_geo_bleu, args, cache_dir):
    """保存checkpoint"""
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint_path = os.path.join(cache_dir, f"checkpoint_epoch{epoch}.pt")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'geo_bleu': geo_bleu,
        'val_geo_bleu': val_geo_bleu,
        'args': vars(args)
    }, checkpoint_path)
    
    return checkpoint_path

# ------------------------------
# 8. 主函数
# ------------------------------

def main():
    args = parse_args()
    
    # 分布式设置
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        print("=" * 80)
        print("LP-BERT X/Y 分离预测训练")
        print("=" * 80)
        print(f"[INFO] 设备: {device}")
        print(f"[INFO] World size: {world_size}")
    
    # 数据预处理
    cache_path = os.path.join(args.cache_dir, "base_data_xy.pkl")
    
    if not os.path.exists(cache_path):
        if rank == 0:
            print("[INFO] 开始数据预处理...")
            
            # 读取原始数据
            df = load_raw_data(args.raw_csv_path)
            
            # 构建序列（X/Y分离）
            user2seq = build_user_sequences_xy(df)
            
            # 划分数据集
            train_users, val_users, test_users = split_train_val_test(user2seq)
            print(f"[INFO] 数据集划分:")
            print(f"  训练集: {len(train_users)} 用户")
            print(f"  验证集: {len(val_users)} 用户")
            print(f"  测试集: {len(test_users)} 用户")
            
            # 生成验证集和测试集
            val_cache_dir = os.path.join(args.cache_dir, "val")
            val_user2input, val_user2target = generate_masked_sequences(
                user2seq, val_users, 60, 74, val_cache_dir, "val"
            )
            
            test_cache_dir = os.path.join(args.cache_dir, "test")
            test_user2input, test_user2target = generate_masked_sequences(
                user2seq, test_users, 60, 74, test_cache_dir, "test"
            )
            
            # 保存预处理数据
            os.makedirs(args.cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({
                    'user2seq': user2seq,
                    'train_users': train_users,
                    'val_users': val_users,
                    'test_users': test_users,
                    'val_user2input': val_user2input,
                    'val_user2target': val_user2target,
                    'test_user2input': test_user2input,
                    'test_user2target': test_user2target
                }, f)
            print("[INFO] 数据预处理完成")
        
        if world_size > 1:
            dist.barrier()
    
    # 加载数据
    if rank == 0:
        print("[INFO] 加载预处理数据...")
    
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    
    user2seq = data['user2seq']
    train_users = data['train_users']
    val_user2input = data['val_user2input']
    val_user2target = data['val_user2target']
    
    # 创建数据集和加载器
    train_dataset = LPBertTrainDatasetXY(user2seq, train_users)
    val_dataset = LPBertValTestDatasetXY(val_user2input, val_user2target)
    
    batch_size_per_gpu = args.batch_size // world_size if world_size > 1 else args.batch_size
    
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=lambda b: collate_fn_train_xy(b, args.mask_days, args.val_mask_prob),
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn_val_test_xy,
        num_workers=2,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"[INFO] 训练批次数: {len(train_loader)}")
        print(f"[INFO] 验证批次数: {len(val_loader)}")
    
    # 创建模型
    model = LPBertModelXY(
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)
    
    # 优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
    
    # DDP包装
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # 检查是否恢复训练
    start_epoch = 0
    if args.resume:
        checkpoint_path, loaded_epoch = find_latest_checkpoint(args.cache_dir)
        if checkpoint_path and os.path.exists(checkpoint_path):
            if rank == 0:
                print(f"[INFO] 加载checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            if rank == 0:
                print(f"[INFO] 从epoch {start_epoch + 1}继续训练")
    
    # 训练循环
    if rank == 0:
        print(f"[INFO] 开始训练...")
    
    best_val_geo_bleu = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        epoch_geo_bleu = 0.0
        num_batches = 0
        num_eval_batches = 0
        
        # 训练进度条
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = train_loader
        
        for batch_idx, batch_data in enumerate(pbar):
            if batch_data is None:
                continue
            
            # 数据到GPU
            for key in batch_data:
                if torch.is_tensor(batch_data[key]):
                    batch_data[key] = batch_data[key].to(device)
            
            # 前向传播
            logits_x, logits_y = model(
                batch_data['d'], batch_data['t'], 
                batch_data['x'], batch_data['y'], 
                batch_data['delta'], batch_data['padding_mask']
            )
            
            # 计算损失
            loss = compute_loss_xy(
                logits_x, logits_y,
                batch_data['target_x'], batch_data['target_y'],
                batch_data['pred_mask']
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # 定期评估
            if batch_idx > 0 and batch_idx % args.eval_interval == 0:
                geo_bleu = evaluate_batch_geo_bleu_xy(model, batch_data, device)
                epoch_geo_bleu += geo_bleu
                num_eval_batches += 1
                
                if rank == 0:
                    avg_loss = epoch_loss / num_batches
                    avg_geo_bleu = epoch_geo_bleu / num_eval_batches
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{avg_loss:.4f}',
                        'geo_bleu': f'{geo_bleu:.4f}',
                        'avg_geo_bleu': f'{avg_geo_bleu:.4f}'
                    })
        
        # Epoch结束
        avg_loss = epoch_loss / max(1, num_batches)
        avg_geo_bleu = epoch_geo_bleu / max(1, num_eval_batches)
        
        # 验证集评估
        if rank == 0:
            print(f"\n[INFO] 在验证集上评估...")
        
        model.eval()
        val_geo_bleu_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="验证", disable=(rank != 0)):
                if batch_data is None:
                    continue
                
                for key in batch_data:
                    if torch.is_tensor(batch_data[key]):
                        batch_data[key] = batch_data[key].to(device)
                
                geo_bleu = evaluate_batch_geo_bleu_xy(model, batch_data, device)
                val_geo_bleu_sum += geo_bleu
                val_batches += 1
        
        val_geo_bleu = val_geo_bleu_sum / max(1, val_batches)
        
        # 调整学习率
        if args.scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(avg_loss)
        
        # 保存checkpoint
        if rank == 0:
            print(f"\n[Epoch {epoch+1}] 训练损失: {avg_loss:.4f}, 训练GEO-BLEU: {avg_geo_bleu:.4f}, 验证GEO-BLEU: {val_geo_bleu:.4f}")
            
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                avg_loss, avg_geo_bleu, val_geo_bleu,
                args, args.cache_dir
            )
            print(f"[INFO] 保存checkpoint: {checkpoint_path}")
            
            # 保存最佳模型
            if val_geo_bleu > best_val_geo_bleu:
                best_val_geo_bleu = val_geo_bleu
                best_path = os.path.join(args.cache_dir, "best_model.pt")
                torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), best_path)
                print(f"[INFO] 保存最佳模型，验证GEO-BLEU: {best_val_geo_bleu:.4f}")
    
    # 清理
    if world_size > 1:
        dist.destroy_process_group()
    
    if rank == 0:
        print("\n[INFO] 训练完成!")
        print(f"[INFO] 最佳验证GEO-BLEU: {best_val_geo_bleu:.4f}")

if __name__ == "__main__":
    main()