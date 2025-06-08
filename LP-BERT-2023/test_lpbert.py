#!/usr/bin/env python3
"""
LP-BERT 测试脚本
直接使用训练代码的结构，测试 GEO-BLEU 和 DTW
"""

import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
import re

# ------------------------------
# 1. 模型定义（与训练代码相同）
# ------------------------------

class LPBertModel(nn.Module):
    def __init__(self, num_dates, num_times, num_locations, max_timedelta, emb_dim=128,
                 max_pos_emb=4096, num_transformer_layers=4, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_locations = num_locations
        self.mask_token_id = num_locations

        # Embedding 层
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

    def forward(self, date_idx, time_idx, loc_idx, delta_idx, padding_mask, token_mask=None):
        if date_idx.numel() == 0:
            return torch.empty((0, self.num_locations), device=date_idx.device)
        
        # 确保数据在正确范围内
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

        # 如果提供了token_mask，只返回masked位置的logits
        if token_mask is not None:
            masked_indices = token_mask.nonzero(as_tuple=False)
            if masked_indices.size(0) == 0:
                return torch.empty((0, self.num_locations), device=x.device)
            batch_idx = masked_indices[:, 0]
            seq_idx = masked_indices[:, 1]
            logits_masked = logits[batch_idx, seq_idx, :]
            return logits_masked
        
        return logits

# ------------------------------
# 2. 评估指标
# ------------------------------

def extract_ngrams(sequence, n):
    """提取序列的n-gram"""
    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i+n])
        ngrams.append(ngram)
    return ngrams

def compute_geo_bleu(pred_sequences, target_sequences, max_n=4):
    """计算Geo-BLEU分数"""
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
        
        bleu_scores.append(geo_mean)
    
    return np.mean(bleu_scores) if bleu_scores else 0.0

def compute_dtw_distance(seq1, seq2):
    """计算两个序列之间的DTW距离"""
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return float('inf')
    
    # 初始化DTW矩阵
    dtw = np.full((n + 1, m + 1), float('inf'))
    dtw[0, 0] = 0
    
    # 填充DTW矩阵
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # 计算位置之间的欧氏距离
            loc1 = seq1[i-1]
            loc2 = seq2[j-1]
            
            # 从location_id恢复x,y坐标 (假设grid_width=200)
            grid_width = 200
            x1, y1 = loc1 // grid_width, loc1 % grid_width
            x2, y2 = loc2 // grid_width, loc2 % grid_width
            
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            # DTW递推
            dtw[i, j] = distance + min(
                dtw[i-1, j],      # 插入
                dtw[i, j-1],      # 删除
                dtw[i-1, j-1]     # 匹配
            )
    
    return dtw[n, m] / max(n, m)  # 归一化

def compute_average_dtw(pred_trajectories, true_trajectories):
    """计算平均DTW距离"""
    dtw_scores = []
    
    for uid in pred_trajectories:
        if uid in true_trajectories:
            pred_traj = pred_trajectories[uid]
            true_traj = true_trajectories[uid]
            
            if len(pred_traj) > 0 and len(true_traj) > 0:
                dtw_score = compute_dtw_distance(pred_traj, true_traj)
                dtw_scores.append(dtw_score)
    
    return np.mean(dtw_scores) if dtw_scores else float('inf')

# ------------------------------
# 3. 数据处理
# ------------------------------

def load_test_data(cache_dir):
    """加载测试数据"""
    print("[INFO] 加载测试数据...")
    
    # 加载base_data
    base_data_path = os.path.join(cache_dir, "base_data.pkl")
    with open(base_data_path, "rb") as f:
        base_data = pickle.load(f)
    
    # 加载测试集数据
    test_input_path = os.path.join(cache_dir, "test/test_user2inputseq.pkl")
    test_target_path = os.path.join(cache_dir, "test/test_user2targetseq.pkl")
    
    with open(test_input_path, "rb") as f:
        test_user2input = pickle.load(f)
    with open(test_target_path, "rb") as f:
        test_user2target = pickle.load(f)
    
    return base_data, test_user2input, test_user2target

def prepare_batch(user_sequences, device, num_locations, max_timedelta=100):
    """准备一个批次的数据"""
    batch_size = len(user_sequences)
    if batch_size == 0:
        return None
    
    # 计算最大长度
    lengths = [len(seq) for seq in user_sequences]
    max_len = max(lengths)
    
    # 创建张量
    date_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
    time_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
    loc_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
    delta_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
    padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    
    for i, seq in enumerate(user_sequences):
        for j, item in enumerate(seq):
            date_tensor[i, j] = item["d"]
            time_tensor[i, j] = item["t"]
            loc_tensor[i, j] = item["loc_id"]
            delta_tensor[i, j] = min(item["delta"], max_timedelta)
            padding_mask[i, j] = True
    
    # 移动到设备
    date_tensor = date_tensor.to(device)
    time_tensor = time_tensor.to(device)
    loc_tensor = loc_tensor.to(device)
    delta_tensor = delta_tensor.to(device)
    padding_mask = padding_mask.to(device)
    
    return date_tensor, time_tensor, loc_tensor, delta_tensor, padding_mask

def predict_sequence(model, input_seq, target_days, device, num_locations):
    """为一个用户预测序列"""
    if len(input_seq) == 0:
        # 如果没有输入序列，返回默认预测
        return [num_locations // 2] * len(target_days)  # 返回中心位置
    
    # 准备输入
    batch_data = prepare_batch([input_seq], device, num_locations)
    if batch_data is None:
        return [num_locations // 2] * len(target_days)
    
    date_tensor, time_tensor, loc_tensor, delta_tensor, padding_mask = batch_data
    
    predictions = []
    
    with torch.no_grad():
        # 获取模型输出
        logits = model(date_tensor, time_tensor, loc_tensor, delta_tensor, padding_mask)
        
        # logits shape: [1, seq_len, num_locations]
        probs = torch.softmax(logits[0], dim=-1)  # [seq_len, num_locations]
        
        # 对每个目标时间点进行预测
        for target_day, target_time in target_days:
            # 简单策略：使用最后几个时间步的平均概率
            if len(input_seq) >= 5:
                avg_probs = probs[-5:].mean(dim=0)
            else:
                avg_probs = probs.mean(dim=0)
            
            # 获取最可能的位置
            pred_loc = torch.argmax(avg_probs).item()
            predictions.append(pred_loc)
    
    return predictions

# ------------------------------
# 4. 主测试函数
# ------------------------------

def find_latest_checkpoint(cache_dir):
    """找到最新的checkpoint"""
    checkpoint_files = []
    pattern = re.compile(r'checkpoint_epoch(\d+)\.pt')
    
    for filename in os.listdir(cache_dir):
        match = pattern.match(filename)
        if match:
            epoch_num = int(match.group(1))
            checkpoint_files.append((filename, epoch_num))
    
    if not checkpoint_files:
        return None
    
    checkpoint_files.sort(key=lambda x: x[1])
    latest_file, latest_epoch = checkpoint_files[-1]
    
    return os.path.join(cache_dir, latest_file), latest_epoch

def test_lpbert(cache_dir='cache_lpbert', checkpoint_path=None, device='cuda'):
    """测试LP-BERT模型"""
    print("=" * 80)
    print("LP-BERT 测试程序")
    print("=" * 80)
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")
    
    # 找到checkpoint
    if checkpoint_path is None:
        checkpoint_path, epoch = find_latest_checkpoint(cache_dir)
        if checkpoint_path is None:
            print(f"[ERROR] 在 {cache_dir} 中找不到checkpoint文件")
            return
        print(f"[INFO] 找到checkpoint: {checkpoint_path} (epoch {epoch})")
    
    # 加载数据
    base_data, test_user2input, test_user2target = load_test_data(cache_dir)
    
    num_locations = base_data['num_locations']
    max_date = base_data['max_date']
    max_time = base_data['max_time']
    max_delta = base_data['max_delta']
    
    print(f"[INFO] 数据统计:")
    print(f"  - 位置数量: {num_locations}")
    print(f"  - 日期范围: 0-{max_date-1}")
    print(f"  - 时间范围: 0-{max_time-1}")
    print(f"  - 测试用户数: {len(test_user2input)}")
    
    # 加载模型
    print("[INFO] 加载模型...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 获取模型配置
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        # 使用默认配置
        model_config = {
            'num_dates': max_date,
            'num_times': max_time,
            'num_locations': num_locations,
            'max_timedelta': max_delta,
            'emb_dim': 128,
            'num_transformer_layers': 4,
            'nhead': 8,
            'dim_feedforward': 512,
            'dropout': 0.1
        }
    
    # 创建模型
    model = LPBertModel(**model_config).to(device)
    
    # 加载权重
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    print(f"[INFO] 模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 预测
    print("[INFO] 开始预测...")
    pred_sequences = {}
    true_sequences = {}
    pred_trajectories = {}
    true_trajectories = {}
    
    for uid in tqdm(test_user2input.keys(), desc="预测进度"):
        input_seq = test_user2input[uid]
        target_seq = test_user2target[uid]
        
        if len(target_seq) == 0:
            continue
        
        # 提取目标时间点
        target_days = [(item["d"], item["t"]) for item in target_seq]
        
        # 预测
        predictions = predict_sequence(model, input_seq, target_days, device, num_locations)
        
        # 提取真实位置
        true_locs = [item["loc_id"] for item in target_seq]
        
        # 保存结果
        pred_sequences[uid] = predictions
        true_sequences[uid] = true_locs
        
        # 用于DTW的完整轨迹
        pred_trajectories[uid] = predictions
        true_trajectories[uid] = true_locs
    
    print(f"[INFO] 预测完成，有效用户数: {len(pred_sequences)}")
    
    # 计算指标
    print("[INFO] 计算评估指标...")
    
    # 将字典转换为列表格式用于GEO-BLEU计算
    pred_list = []
    true_list = []
    for uid in pred_sequences:
        pred_list.append(pred_sequences[uid])
        true_list.append(true_sequences[uid])
    
    # 计算GEO-BLEU
    geo_bleu = compute_geo_bleu(pred_list, true_list, max_n=4)
    
    # 计算DTW
    dtw = compute_average_dtw(pred_trajectories, true_trajectories)
    
    # 输出结果
    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)
    print(f"GEO-BLEU: {geo_bleu:.4f}")
    print(f"DTW: {dtw:.4f}")
    print("\n与论文结果对比:")
    print(f"论文 GEO-BLEU: 0.3440")
    print(f"论文 DTW: 29.9633")
    print(f"\n相对性能:")
    print(f"GEO-BLEU 比例: {geo_bleu/0.3440:.2%}")
    if dtw < 29.9633:
        print(f"DTW 改进: {(29.9633-dtw)/29.9633:.2%}")
    else:
        print(f"DTW 差距: {(dtw-29.9633)/29.9633:.2%}")
    
    # 保存结果
    results = {
        'geo_bleu': geo_bleu,
        'dtw': dtw,
        'checkpoint': checkpoint_path,
        'num_test_users': len(pred_sequences),
        'model_config': model_config
    }
    
    result_path = os.path.join(cache_dir, 'test_results.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n[INFO] 结果已保存到: {result_path}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试LP-BERT模型")
    parser.add_argument('--cache_dir', type=str, default='cache_lpbert', help='缓存目录')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    test_lpbert(args.cache_dir, args.checkpoint, args.device)