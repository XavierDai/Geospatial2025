#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_data_prep.py

简化的2024年数据预处理 - 直接用于SF-BERT + CM-BERT
不做复杂的格式转换，保持原始2024年格式
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def load_city_data(data_dir="../Data2024"):
    """直接加载城市数据 - 跳过城市A"""
    data_path = Path(data_dir)
    cities_data = {}
    
    print("[INFO] 加载城市数据（跳过城市A）...")
    
    # 只处理 B、C、D 城市
    for city in ['B', 'C', 'D']:
        csv_file = data_path / f"city{city}-dataset.csv"
        if csv_file.exists():
            print(f"[INFO] 读取城市{city}: {csv_file}")
            
            # 直接读取，不做复杂转换
            df = pd.read_csv(csv_file)
            print(f"[INFO] 城市{city}: {len(df):,} 记录, {df['uid'].nunique():,} 用户")
            
            # 简单的用户序列构建 - 快速版本
            print(f"[INFO] 构建城市{city}用户序列...")
            user_sequences = build_simple_sequences(df)
            
            cities_data[city] = {
                'dataframe': df,
                'user_sequences': user_sequences,
                'stats': {
                    'total_records': len(df),
                    'unique_users': df['uid'].nunique(),
                    'date_range': (df['d'].min(), df['d'].max()),
                    'time_range': (df['t'].min(), df['t'].max()),
                    'x_range': (df['x'].min(), df['x'].max()),
                    'y_range': (df['y'].min(), df['y'].max())
                }
            }
            
            print(f"[SUCCESS] 城市{city}处理完成")
    
    return cities_data

def build_simple_sequences(df):
    """快速构建用户序列 - 保持原始2024格式"""
    user_sequences = {}
    
    # 按用户分组
    for uid, group in tqdm(df.groupby('uid'), desc="构建序列"):
        # 按时间排序
        group_sorted = group.sort_values(['d', 't']).reset_index(drop=True)
        
        # 简单计算时间差
        time_diffs = [0] + [
            min(47, max(0, 
                ((row.d - prev_row.d) * 48 + (row.t - prev_row.t))
            ))
            for prev_row, row in zip(group_sorted.itertuples(), group_sorted.iloc[1:].itertuples())
        ]
        
        # 构建序列
        sequence = []
        for i, row in group_sorted.iterrows():
            sequence.append({
                'd': int(row['d']),      # 0-74 (保持原始)
                't': int(row['t']),      # 0-47 (保持原始)
                'x': int(row['x']),      # 1-200
                'y': int(row['y']),      # 1-200
                'delta': time_diffs[i]
            })
        
        user_sequences[uid] = sequence
    
    return user_sequences

def calculate_stay_frequency(user_sequences):
    """计算停留频率模式 - 论文核心功能"""
    print("[INFO] 计算停留频率模式...")
    
    # 时间段划分
    time_segments = {
        0: list(range(0, 12)),    # morning: 0-11
        1: list(range(12, 18)),   # daytime: 12-17  
        2: list(range(18, 36)),   # evening: 18-35
        3: list(range(36, 48))    # night: 36-47
    }
    
    def get_time_segment(t):
        for seg_id, times in time_segments.items():
            if t in times:
                return seg_id
        return 0
    
    def is_weekday(d):
        return (d % 7) not in [0, 6]  # 简化版本
    
    user_stay_patterns = {}
    
    for uid, sequence in tqdm(user_sequences.items(), desc="计算停留频率"):
        # 统计区域访问频率
        area_visits = defaultdict(lambda: defaultdict(list))
        
        for record in sequence:
            area_id = record['x'] * 1000 + record['y']  # 简单区域编码
            time_seg = get_time_segment(record['t'])
            weekday = is_weekday(record['d'])
            day_type = 'weekday' if weekday else 'weekend'
            
            area_visits[area_id][(time_seg, day_type)].append(record['d'])
        
        # 为每条记录计算频率类别
        pattern = []
        for record in sequence:
            area_id = record['x'] * 1000 + record['y']
            time_seg = get_time_segment(record['t'])
            weekday = is_weekday(record['d'])
            day_type = 'weekday' if weekday else 'weekend'
            
            # 计算访问频率
            visits = area_visits[area_id].get((time_seg, day_type), [])
            total_possible = max(1, len([d for d in range(record['d']) if is_weekday(d) == weekday]))
            frequency = len(set(visits)) / total_possible
            
            # 频率分类 (0-3)
            if frequency <= 0.1:
                freq_class = 0
            elif frequency <= 0.2:
                freq_class = 1
            elif frequency <= 0.4:
                freq_class = 2
            else:
                freq_class = 3
            
            pattern.append({
                'd': record['d'],
                't': record['t'],
                'x': record['x'],
                'y': record['y'],
                'delta': record['delta'],
                'time_segment': time_seg,
                'is_weekday': weekday,
                'frequency_class': freq_class
            })
        
        user_stay_patterns[uid] = pattern
    
    return user_stay_patterns

def save_processed_data(cities_data, cities_stay_patterns, output_dir="./processed_data_simple"):
    """保存处理后的数据"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"[INFO] 保存数据到: {output_path}")
    
    for city, data in cities_data.items():
        print(f"[INFO] 保存城市{city}...")
        
        # 保存用户序列
        seq_file = output_path / f"city_{city}_sequences.pkl"
        with open(seq_file, 'wb') as f:
            pickle.dump(data['user_sequences'], f)
        
        # 保存停留频率模式
        if city in cities_stay_patterns:
            freq_file = output_path / f"city_{city}_stay_patterns.pkl"
            with open(freq_file, 'wb') as f:
                pickle.dump(cities_stay_patterns[city], f)
        
        # 保存统计信息
        stats_file = output_path / f"city_{city}_stats.txt"
        with open(stats_file, 'w') as f:
            stats = data['stats']
            f.write(f"城市 {city} 数据统计\n")
            f.write(f"================\n")
            f.write(f"总记录数: {stats['total_records']:,}\n")
            f.write(f"唯一用户数: {stats['unique_users']:,}\n")
            f.write(f"日期范围: {stats['date_range'][0]} - {stats['date_range'][1]}\n")
            f.write(f"时间范围: {stats['time_range'][0]} - {stats['time_range'][1]}\n")
            f.write(f"X坐标范围: {stats['x_range'][0]} - {stats['x_range'][1]}\n")
            f.write(f"Y坐标范围: {stats['y_range'][0]} - {stats['y_range'][1]}\n")
            
            # 序列长度统计
            seq_lengths = [len(seq) for seq in data['user_sequences'].values()]
            f.write(f"\n序列长度统计:\n")
            f.write(f"平均序列长度: {np.mean(seq_lengths):.2f}\n")
            f.write(f"最短序列长度: {min(seq_lengths)}\n")
            f.write(f"最长序列长度: {max(seq_lengths)}\n")
    
    print(f"[SUCCESS] 数据保存完成！")

def split_train_test_users(cities_data):
    """划分训练和测试用户 - 仅处理B、C、D城市"""
    print("[INFO] 划分训练和测试用户...")
    
    split_info = {}
    
    for city, data in cities_data.items():
        all_users = list(data['user_sequences'].keys())
        
        # B、C、D城市：最后3000个用户用于测试（如果有的话）
        if len(all_users) >= 3000:
            test_users = all_users[-3000:]
            train_users = all_users[:-3000]
        else:
            # 如果用户不足3000，取一半
            split_point = len(all_users) // 2
            test_users = all_users[split_point:]
            train_users = all_users[:split_point]
        
        split_info[city] = {
            'train_users': train_users,
            'test_users': test_users,
            'total_users': len(all_users)
        }
        
        print(f"[INFO] 城市{city}: 总用户{len(all_users)}, 训练{len(train_users)}, 测试{len(test_users)}")
    
    return split_info

def main():
    """主函数"""
    print("简化版 HuMob 2024 数据预处理 (B、C、D城市)")
    print("=" * 50)
    print("[INFO] 跳过城市A，仅处理B、C、D城市")
    print("[INFO] 直接使用2024年数据格式，无需转换")
    
    # 1. 加载数据
    cities_data = load_city_data()
    if not cities_data:
        print("[ERROR] 没有找到城市数据")
        return
    
    # 2. 计算停留频率模式
    print("\n[INFO] 计算停留频率模式...")
    cities_stay_patterns = {}
    for city, data in cities_data.items():
        print(f"[INFO] 处理城市{city}...")
        stay_patterns = calculate_stay_frequency(data['user_sequences'])
        cities_stay_patterns[city] = stay_patterns
    
    # 3. 划分训练测试用户
    split_info = split_train_test_users(cities_data)
    
    # 4. 保存数据
    save_processed_data(cities_data, cities_stay_patterns)
    
    # 5. 保存用户划分信息
    with open("./processed_data_simple/user_splits.pkl", 'wb') as f:
        pickle.dump(split_info, f)
    
    print(f"\n[SUCCESS] 数据预处理完成！")
    print(f"[INFO] 处理的城市: B、C、D")
    print(f"[INFO] 数据格式:")
    print(f"  - 日期: 0-74 (原始格式)")
    print(f"  - 时间: 0-47 (原始格式)")
    print(f"  - 坐标: 1-200")
    print(f"[INFO] 输出目录: ./processed_data_simple/")
    print(f"[INFO] 下一步运行 SF-BERT + CM-BERT 训练 (使用B、C、D城市)")
    print(f"[INFO] 注意：SF-BERT将使用B、C、D的混合数据进行跨城市学习")

if __name__ == "__main__":
    main()