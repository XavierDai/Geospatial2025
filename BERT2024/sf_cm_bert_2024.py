#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sf_cm_bert_2024.py

SF-BERT + CM-BERT 实现 (兼容2023年代码风格)
基于论文: "Time-series Stay Frequency for Multi-City Next Location Prediction using Multiple BERTs"

使用方法:
1. 先运行 data_preparation.py 预处理数据
2. 运行此脚本进行完整的训练流程
"""

import os
import json
import pickle
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict, Counter

# 重用2023年的评估函数
def calc_distance(point1, point2, scale_factor=1.):
    """计算两点之间的欧氏距离"""
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]
    return np.sqrt(x_diff ** 2. + y_diff ** 2.) / scale_factor

def calc_point_proximity(point1, point2, beta):
    """计算两点之间的邻近性得分"""
    distance = calc_distance(point1, point2)
    return np.exp(-beta * distance)

def calc_ngram_proximity(ngram1, ngram2, beta):
    """计算两个n-gram之间的邻近性得分"""
    point_proximity_list = list()
    for point1, point2 in zip(ngram1, ngram2):
        point_proximity_list.append(calc_point_proximity(point1, point2, beta))
    return np.prod(point_proximity_list)

def gen_ngram_list(seq, n):
    """生成n-gram列表"""
    if len(seq) < n:
        return []
    return [seq[i:i + n] for i in range(len(seq) - n + 1)]

def calc_geo_p_n(sys_seq, ans_seq, n, beta):
    """计算Geo-BLEU的p_n值"""
    sys_ngram_list = gen_ngram_list(sys_seq, n)
    ans_ngram_list = gen_ngram_list(ans_seq, n)

    edge_list = []
    for sys_id, sys_ngram in enumerate(sys_ngram_list):
        for ans_id, ans_ngram in enumerate(ans_ngram_list):
            proximity = calc_ngram_proximity(sys_ngram, ans_ngram, beta)
            edge_list.append(((sys_id, ans_id), proximity))
   
    edge_list.sort(key=lambda x: x[1], reverse=True)
    proximity_sum = 0.
    proximity_cnt = 0

    while edge_list:
        best_edge = edge_list[0]
        proximity = best_edge[1]
        best_sys_id, best_ans_id = best_edge[0]
        
        proximity_sum += proximity
        proximity_cnt += 1
        
        edge_list = [edge for edge in edge_list[1:] 
                    if edge[0][0] != best_sys_id and edge[0][1] != best_ans_id]
    
    return proximity_sum / max(1, proximity_cnt)

def calc_geobleu(sys_seq, ans_seq, max_n=3, beta=0.5):
    """计算Geo-BLEU"""
    if not sys_seq or not ans_seq:
        return 0.0
        
    p_n_list = []
    seq_len_min = min(len(sys_seq), len(ans_seq))
    max_n_alt = min(max_n, seq_len_min)

    for i in range(1, max_n_alt + 1):
        p_n = calc_geo_p_n(sys_seq, ans_seq, i, beta)
        p_n_list.append(p_n)
    
    if not p_n_list:
        return 0.0
        
    brevity_penalty = 1. if len(sys_seq) > len(ans_seq) else np.exp(1. - len(ans_seq) / len(sys_seq))
    
    # 几何平均
    if all(p > 0 for p in p_n_list):
        geo_mean = np.power(np.prod(p_n_list), 1.0 / len(p_n_list))
    else:
        geo_mean = 0.0
    
    return brevity_penalty * geo_mean

# ------------------------------
# 1. 停留频率计算
# ------------------------------

class StayFrequencyCalculator:
    """时间序列停留频率计算器"""
    
    def __init__(self):
        # 时间段划分 (基于论文Table 2)
        self.time_segments = [
            list(range(0, 12)),     # 0-11: morning
            list(range(12, 18)),    # 12-17: daytime  
            list(range(18, 36)),    # 18-35: evening
            list(range(36, 48))     # 36-47: night
        ]
        
    def calculate_stay_frequency(self, user_sequences):
        """计算用户停留频率模式"""
        print("[INFO] 计算停留频率模式...")
        
        user_stay_patterns = {}
        for uid, sequence in tqdm(user_sequences.items(), desc="计算停留频率"):
            stay_pattern = self._calculate_user_pattern(sequence)
            user_stay_patterns[uid] = stay_pattern
        
        return user_stay_patterns
    
    def _calculate_user_pattern(self, sequence):
        """计算单个用户的停留频率模式"""
        # 统计每个区域在不同时间段的访问频率
        area_visits = defaultdict(lambda: defaultdict(list))
        
        for record in sequence:
            day = record['d']
            time = record['t']
            x, y = record['x'], record['y']
            
            time_segment = self._get_time_segment(time)
            is_weekday = self._is_weekday(day)
            area_id = x * 1000 + y  # 简单的区域编码
            
            day_type = 'weekday' if is_weekday else 'weekend'
            area_visits[area_id][(time_segment, day_type)].append(day)
        
        # 为每个记录计算频率类别
        pattern = []
        for record in sequence:
            day = record['d']
            time = record['t']
            x, y = record['x'], record['y']
            
            time_segment = self._get_time_segment(time)
            is_weekday = self._is_weekday(day)
            area_id = x * 1000 + y
            day_type = 'weekday' if is_weekday else 'weekend'
            
            # 计算该区域在该时间段类型的访问频率
            visit_days = area_visits[area_id].get((time_segment, day_type), [])
            total_possible_days = self._count_day_type_before(day, is_weekday)
            
            frequency = len(set(visit_days)) / max(1, total_possible_days)
            frequency_class = self._classify_frequency(frequency)
            
            pattern.append({
                'd': day,
                't': time,
                'x': x,
                'y': y,
                'delta': record.get('delta', 0),
                'time_segment': time_segment,
                'is_weekday': is_weekday,
                'frequency_class': frequency_class
            })
        
        return pattern
    
    def _get_time_segment(self, time_idx):
        """获取时间段 (0-3)"""
        for i, segment in enumerate(self.time_segments):
            if time_idx in segment:
                return i
        return 0
    
    def _is_weekday(self, day):
        """判断是否为工作日"""
        return (day % 7) not in [0, 6]  # 简化版本
    
    def _count_day_type_before(self, current_day, is_weekday):
        """计算当前日期之前的同类型天数"""
        count = 0
        for d in range(1, current_day):
            if self._is_weekday(d) == is_weekday:
                count += 1
        return max(1, count)
    
    def _classify_frequency(self, frequency):
        """将频率分类为0-3"""
        if frequency <= 0.1:
            return 0
        elif frequency <= 0.2:
            return 1
        elif frequency <= 0.4:
            return 2
        else:
            return 3

# ------------------------------
# 2. 模型定义
# ------------------------------

class SFBert(nn.Module):
    """SF-BERT: 学习跨城市的时间序列停留频率模式"""
    
    def __init__(self, emb_dim=128, num_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        
        # Embedding层
        self.freq_emb = nn.Embedding(5, emb_dim)       # 频率类别 0-3 + mask(4)
        self.time_seg_emb = nn.Embedding(4, emb_dim)   # 时间段 0-3
        self.weekday_emb = nn.Embedding(2, emb_dim)    # 工作日/周末
        self.date_emb = nn.Embedding(76, emb_dim)      # 日期 1-75
        self.time_emb = nn.Embedding(49, emb_dim)      # 时间 0-47
        self.delta_emb = nn.Embedding(48, emb_dim)     # 时间间隔
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 预测头
        self.freq_head = nn.Linear(emb_dim, 4)
        
    def forward(self, freq_classes, time_segments, weekdays, dates, times, deltas, padding_mask):
        # Embedding
        freq_e = self.freq_emb(freq_classes)
        time_seg_e = self.time_seg_emb(time_segments)
        weekday_e = self.weekday_emb(weekdays)
        date_e = self.date_emb(dates)
        time_e = self.time_emb(times)
        delta_e = self.delta_emb(deltas)
        
        # 特征融合
        x = freq_e + time_seg_e + weekday_e + date_e + time_e + delta_e
        
        # Transformer
        src_key_padding_mask = ~padding_mask
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # 预测
        freq_logits = self.freq_head(out)
        
        return out, freq_logits

class CMBert(nn.Module):
    """CM-BERT: 学习城市特定的移动模式"""
    
    def __init__(self, emb_dim=128, num_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        
        # Embedding层
        self.x_emb = nn.Embedding(202, emb_dim)        # X坐标 1-200 + mask(201)
        self.y_emb = nn.Embedding(202, emb_dim)        # Y坐标 1-200 + mask(201)
        self.date_emb = nn.Embedding(76, emb_dim)      # 日期
        self.time_emb = nn.Embedding(49, emb_dim)      # 时间
        self.delta_emb = nn.Embedding(48, emb_dim)     # 时间间隔
        self.freq_emb = nn.Embedding(4, emb_dim)       # 频率类别
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 预测头
        self.x_head = nn.Linear(emb_dim, 200)
        self.y_head = nn.Linear(emb_dim, 200)
        
    def forward(self, x_coords, y_coords, dates, times, deltas, freq_classes, padding_mask):
        # Embedding
        x_e = self.x_emb(x_coords)
        y_e = self.y_emb(y_coords)
        date_e = self.date_emb(dates)
        time_e = self.time_emb(times)
        delta_e = self.delta_emb(deltas)
        freq_e = self.freq_emb(freq_classes)
        
        # 特征融合
        x = x_e + y_e + date_e + time_e + delta_e + freq_e
        
        # Transformer
        src_key_padding_mask = ~padding_mask
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # 预测
        x_logits = self.x_head(out)
        y_logits = self.y_head(out)
        
        return out, x_logits, y_logits

class LPBert(nn.Module):
    """LP-BERT: 最终位置预测，整合SF-BERT和CM-BERT特征"""
    
    def __init__(self, emb_dim=128, num_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        
        # 位置和时间embedding
        self.x_emb = nn.Embedding(202, emb_dim)
        self.y_emb = nn.Embedding(202, emb_dim)  
        self.date_emb = nn.Embedding(76, emb_dim)
        self.time_emb = nn.Embedding(49, emb_dim)
        self.delta_emb = nn.Embedding(48, emb_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 预测头
        self.x_head = nn.Linear(emb_dim, 200)
        self.y_head = nn.Linear(emb_dim, 200)
    
    def forward(self, x_coords, y_coords, dates, times, deltas, 
                sf_features, cm_features, padding_mask):
        # 基础embedding
        x_e = self.x_emb(x_coords)
        y_e = self.y_emb(y_coords)
        date_e = self.date_emb(dates)
        time_e = self.time_emb(times)
        delta_e = self.delta_emb(deltas)
        
        # 论文中的特征融合方式：向量相加
        x = x_e + y_e + date_e + time_e + delta_e + sf_features + cm_features
        
        # Transformer
        src_key_padding_mask = ~padding_mask
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # 预测
        x_logits = self.x_head(out)
        y_logits = self.y_head(out)
        
        return x_logits, y_logits

# ------------------------------
# 3. 数据集和训练函数
# ------------------------------

def load_city_data(data_dir="./processed_data"):
    """加载城市数据"""
    data_path = Path(data_dir)
    cities_data = {}
    
    for city in ['A', 'B', 'C', 'D']:
        seq_file = data_path / f"city_{city}_user2seq.pkl"
        if seq_file.exists():
            with open(seq_file, 'rb') as f:
                user2seq = pickle.load(f)
            cities_data[city] = user2seq
            print(f"[INFO] 加载城市{city}: {len(user2seq)} 用户")
    
    return cities_data

def prepare_datasets(cities_data, cities_stay_patterns):
    """准备训练和测试数据集"""
    print("[INFO] 准备数据集...")
    
    # 分离训练用户和测试用户 (基于2024年任务设定)
    datasets = {}
    
    for city in ['A', 'B', 'C', 'D']:
        if city not in cities_data:
            continue
            
        user_sequences = cities_data[city]
        stay_patterns = cities_stay_patterns[city]
        
        # 获取所有用户ID
        all_users = list(user_sequences.keys())
        
        if city == 'A':
            # 城市A不需要预测，全部用于训练
            train_users = all_users
            test_users = []
        else:
            # 其他城市：选择3000个用户用于测试
            test_users = all_users[-3000:] if len(all_users) >= 3000 else all_users[-len(all_users)//2:]
            train_users = [u for u in all_users if u not in test_users]
        
        datasets[city] = {
            'train_users': train_users,
            'test_users': test_users,
            'user_sequences': user_sequences,
            'stay_patterns': stay_patterns
        }
        
        print(f"[INFO] 城市{city}: 训练用户{len(train_users)}, 测试用户{len(test_users)}")
    
    return datasets

def collate_fn_simple(batch, mode='sf'):
    """简化的collate函数"""
    max_len = 512  # 限制最大长度
    batch_size = len(batch)
    
    # 截取序列
    sequences = []
    for seq in batch:
        if len(seq) > max_len:
            start_idx = random.randint(0, len(seq) - max_len)
            sequences.append(seq[start_idx:start_idx + max_len])
        else:
            sequences.append(seq)
    
    # 找最大长度
    actual_max_len = max(len(seq) for seq in sequences)
    
    if mode == 'sf':
        # SF-BERT所需的张量
        freq_classes = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        time_segments = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        weekdays = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        dates = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        times = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        deltas = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        padding_mask = torch.zeros(batch_size, actual_max_len, dtype=torch.bool)
        pred_mask = torch.zeros(batch_size, actual_max_len, dtype=torch.bool)
        target_freq = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            mask_positions = random.sample(range(seq_len), min(100, seq_len))
            
            for j, record in enumerate(seq):
                freq_classes[i, j] = record['frequency_class']
                time_segments[i, j] = record['time_segment']
                weekdays[i, j] = int(record['is_weekday'])
                dates[i, j] = record['d']
                times[i, j] = record['t']
                deltas[i, j] = record['delta']
                padding_mask[i, j] = True
                
                if j in mask_positions:
                    target_freq[i, j] = record['frequency_class']
                    freq_classes[i, j] = 4  # mask token
                    pred_mask[i, j] = True
        
        return {
            'freq_classes': freq_classes,
            'time_segments': time_segments,
            'weekdays': weekdays,
            'dates': dates,
            'times': times,
            'deltas': deltas,
            'padding_mask': padding_mask,
            'pred_mask': pred_mask,
            'target_freq': target_freq
        }
    
    elif mode == 'cm':
        # CM-BERT所需的张量
        x_coords = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        y_coords = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        dates = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        times = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        deltas = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        freq_classes = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        padding_mask = torch.zeros(batch_size, actual_max_len, dtype=torch.bool)
        pred_mask = torch.zeros(batch_size, actual_max_len, dtype=torch.bool)
        target_x = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        target_y = torch.zeros(batch_size, actual_max_len, dtype=torch.long)
        
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            mask_positions = random.sample(range(seq_len), min(100, seq_len))
            
            for j, record in enumerate(seq):
                x_coords[i, j] = record['x']
                y_coords[i, j] = record['y']
                dates[i, j] = record['d']
                times[i, j] = record['t']
                deltas[i, j] = record['delta']
                freq_classes[i, j] = record['frequency_class']
                padding_mask[i, j] = True
                
                if j in mask_positions:
                    target_x[i, j] = record['x'] - 1  # 转0-based
                    target_y[i, j] = record['y'] - 1
                    x_coords[i, j] = 201  # mask token
                    y_coords[i, j] = 201
                    pred_mask[i, j] = True
        
        return {
            'x_coords': x_coords,
            'y_coords': y_coords,
            'dates': dates,
            'times': times,
            'deltas': deltas,
            'freq_classes': freq_classes,
            'padding_mask': padding_mask,
            'pred_mask': pred_mask,
            'target_x': target_x,
            'target_y': target_y
        }

class SimpleDataset(Dataset):
    """简化的数据集类"""
    
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def train_model(model, train_loader, device, epochs=20, lr=2e-5, model_name="model"):
    """通用训练函数"""
    print(f"\n[INFO] 开始训练 {model_name}...")
    
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}")
        
        for batch_data in pbar:
            if batch_data is None:
                continue
                
            # 数据移到设备
            for key in batch_data:
                if torch.is_tensor(batch_data[key]):
                    batch_data[key] = batch_data[key].to(device)
            
            # 前向传播
            if model_name == "SF-BERT":
                features, logits = model(
                    batch_data['freq_classes'],
                    batch_data['time_segments'],
                    batch_data['weekdays'],
                    batch_data['dates'],
                    batch_data['times'],
                    batch_data['deltas'],
                    batch_data['padding_mask']
                )
                
                # 计算损失
                mask_indices = batch_data['pred_mask'].nonzero(as_tuple=False)
                if mask_indices.size(0) > 0:
                    batch_idx = mask_indices[:, 0]
                    seq_idx = mask_indices[:, 1]
                    pred_logits = logits[batch_idx, seq_idx]
                    true_labels = batch_data['target_freq'][batch_idx, seq_idx]
                    loss = criterion(pred_logits, true_labels)
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                    
            elif model_name == "CM-BERT":
                features, x_logits, y_logits = model(
                    batch_data['x_coords'],
                    batch_data['y_coords'],
                    batch_data['dates'],
                    batch_data['times'],
                    batch_data['deltas'],
                    batch_data['freq_classes'],
                    batch_data['padding_mask']
                )
                
                # 计算损失
                mask_indices = batch_data['pred_mask'].nonzero(as_tuple=False)
                if mask_indices.size(0) > 0:
                    batch_idx = mask_indices[:, 0]
                    seq_idx = mask_indices[:, 1]
                    
                    pred_x = x_logits[batch_idx, seq_idx]
                    pred_y = y_logits[batch_idx, seq_idx]
                    true_x = batch_data['target_x'][batch_idx, seq_idx]
                    true_y = batch_data['target_y'][batch_idx, seq_idx]
                    
                    valid_mask = (true_x >= 0) & (true_y >= 0)
                    if valid_mask.sum() > 0:
                        loss_x = criterion(pred_x[valid_mask], true_x[valid_mask])
                        loss_y = criterion(pred_y[valid_mask], true_y[valid_mask])
                        loss = loss_x + loss_y
                    else:
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"[{model_name}] Epoch {epoch+1}: 平均损失 = {avg_loss:.4f}")
    
    print(f"[SUCCESS] {model_name} 训练完成！")
    return model

def main():
    """主函数"""
    print("SF-BERT + CM-BERT 2024年版本")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")
    
    # 1. 加载数据
    print("\n[INFO] 加载数据...")
    cities_data = load_city_data()
    if not cities_data:
        print("[ERROR] 请先运行 data_preparation.py")
        return
    
    # 2. 计算停留频率
    print("\n[INFO] 计算停留频率...")
    freq_calculator = StayFrequencyCalculator()
    cities_stay_patterns = {}
    
    for city, user_sequences in cities_data.items():
        print(f"[INFO] 计算城市{city}的停留频率...")
        stay_patterns = freq_calculator.calculate_stay_frequency(user_sequences)
        cities_stay_patterns[city] = stay_patterns
    
    # 3. 准备数据集
    datasets = prepare_datasets(cities_data, cities_stay_patterns)
    
    # 4. 训练SF-BERT (跨城市)
    print("\n[INFO] 准备SF-BERT训练数据...")
    sf_sequences = []
    for city, data in datasets.items():
        for uid in data['train_users']:
            if uid in data['stay_patterns']:
                sf_sequences.append(data['stay_patterns'][uid])
    
    sf_dataset = SimpleDataset(sf_sequences)
    sf_loader = DataLoader(
        sf_dataset, batch_size=8, shuffle=True,
        collate_fn=lambda b: collate_fn_simple(b, mode='sf'),
        num_workers=2
    )
    
    sf_bert = SFBert()
    sf_bert = train_model(sf_bert, sf_loader, device, epochs=20, model_name="SF-BERT")
    
    # 保存SF-BERT
    torch.save(sf_bert.state_dict(), "./sf_bert.pt")
    print("[INFO] SF-BERT已保存")
    
    # 5. 训练CM-BERT (每个城市)
    cm_berts = {}
    for city, data in datasets.items():
        print(f"\n[INFO] 训练城市{city}的CM-BERT...")
        
        cm_sequences = []
        for uid in data['train_users']:
            if uid in data['stay_patterns']:
                cm_sequences.append(data['stay_patterns'][uid])
        
        if not cm_sequences:
            continue
            
        cm_dataset = SimpleDataset(cm_sequences)
        cm_loader = DataLoader(
            cm_dataset, batch_size=8, shuffle=True,
            collate_fn=lambda b: collate_fn_simple(b, mode='cm'),
            num_workers=2
        )
        
        cm_bert = CMBert()
        cm_bert = train_model(cm_bert, cm_loader, device, epochs=20, model_name=f"CM-BERT-{city}")
        cm_berts[city] = cm_bert
        
        # 保存CM-BERT
        torch.save(cm_bert.state_dict(), f"./cm_bert_{city}.pt")
        print(f"[INFO] CM-BERT-{city}已保存")
    
    print("\n[SUCCESS] 所有模型训练完成！")
    print("[INFO] 下一步可以训练LP-BERT进行最终预测")

if __name__ == "__main__":
    main()
