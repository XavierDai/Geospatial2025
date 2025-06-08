#!/usr/bin/env python3
"""
暴力完整分析HuMob数据集 - 不搞花里胡哨的
"""

import pandas as pd
import numpy as np
import os
from collections import Counter, defaultdict
import time

def brute_force_analysis(file_path):
    """
    暴力分析整个文件 - 直接扫描所有数据
    """
    print("="*60)
    print("暴力完整分析 - 扫描整个文件")
    print("="*60)
    
    start_time = time.time()
    
    # 文件基本信息
    file_size = os.path.getsize(file_path)
    print(f"文件大小: {file_size / (1024**3):.2f} GB")
    
    # 初始化统计变量
    total_rows = 0
    all_users = set()
    all_dates = set()
    all_times = set()
    all_x = set()
    all_y = set()
    mask_count = 0
    mask_users = set()
    
    # 用户记录数统计
    user_record_counts = defaultdict(int)
    
    # 按天统计用户
    users_by_day = defaultdict(set)
    
    # 掩码数据统计
    mask_by_user = defaultdict(int)
    mask_by_day = defaultdict(int)
    
    print("\n开始暴力扫描...")
    print("进度显示: 每处理100万行显示一次")
    
    chunk_size = 100000
    chunk_count = 0
    
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk_count += 1
            total_rows += len(chunk)
            
            # 统计基本信息
            all_users.update(chunk['uid'].unique())
            all_dates.update(chunk['d'].unique())
            all_times.update(chunk['t'].unique())
            all_x.update(chunk['x'].unique())
            all_y.update(chunk['y'].unique())
            
            # 统计每个用户的记录数
            for uid, count in chunk['uid'].value_counts().items():
                user_record_counts[uid] += count
            
            # 按天统计用户
            for _, row in chunk.groupby('d')['uid'].apply(set).items():
                day = _
                users_by_day[day].update(row)
            
            # 检查掩码数据
            mask_rows = chunk[(chunk['x'] == 999) | (chunk['y'] == 999)]
            mask_count += len(mask_rows)
            
            if len(mask_rows) > 0:
                mask_users.update(mask_rows['uid'].unique())
                
                # 按用户统计掩码
                for uid, count in mask_rows['uid'].value_counts().items():
                    mask_by_user[uid] += count
                
                # 按天统计掩码
                for day, count in mask_rows['d'].value_counts().items():
                    mask_by_day[day] += count
            
            # 显示进度
            if total_rows % 1000000 == 0:
                elapsed = time.time() - start_time
                print(f"   已处理 {total_rows:,} 行, 发现 {len(all_users):,} 个用户, 用时 {elapsed:.1f}秒")
    
    except Exception as e:
        print(f"扫描过程中出错: {e}")
        return
    
    # 计算总用时
    total_time = time.time() - start_time
    
    print(f"\n扫描完成! 总用时: {total_time:.1f}秒")
    print("="*60)
    print("完整统计结果")
    print("="*60)
    
    # 基本统计
    print(f"总行数: {total_rows:,}")
    print(f"总用户数: {len(all_users):,}")
    print(f"用户ID范围: {min(all_users)} - {max(all_users)}")
    print(f"总天数: {len(all_dates)} (范围: {min(all_dates)} - {max(all_dates)})")
    print(f"总时间段数: {len(all_times)} (范围: {min(all_times)} - {max(all_times)})")
    print(f"X坐标范围: {min(all_x)} - {max(all_x)} (共{len(all_x)}个不同值)")
    print(f"Y坐标范围: {min(all_y)} - {max(all_y)} (共{len(all_y)}个不同值)")
    print(f"总位置数: {len(all_x) * len(all_y)} (理论最大)")
    
    # 掩码数据统计
    print(f"\n掩码数据统计:")
    print(f"掩码记录总数: {mask_count:,}")
    print(f"掩码记录比例: {(mask_count/total_rows)*100:.2f}%")
    print(f"有掩码的用户数: {len(mask_users)}")
    
    if mask_users:
        print(f"掩码用户ID范围: {min(mask_users)} - {max(mask_users)}")
        
        # 掩码最多的用户
        top_mask_users = sorted(mask_by_user.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"掩码记录最多的10个用户:")
        for uid, count in top_mask_users:
            print(f"  用户{uid}: {count}条掩码记录")
        
        # 掩码最多的天
        top_mask_days = sorted(mask_by_day.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"掩码记录最多的10天:")
        for day, count in top_mask_days:
            print(f"  第{day}天: {count}条掩码记录")
    
    # 用户记录数统计
    record_counts = list(user_record_counts.values())
    print(f"\n用户记录数统计:")
    print(f"平均每用户记录数: {np.mean(record_counts):.1f}")
    print(f"记录数中位数: {np.median(record_counts):.1f}")
    print(f"最多记录数: {max(record_counts)}")
    print(f"最少记录数: {min(record_counts)}")
    
    # 每天的用户数统计
    daily_user_counts = [(day, len(users)) for day, users in users_by_day.items()]
    daily_user_counts.sort()
    
    print(f"\n每天活跃用户数统计:")
    print(f"前10天的用户数:")
    for day, count in daily_user_counts[:10]:
        print(f"  第{day}天: {count:,}个用户")
    
    print(f"后10天的用户数:")
    for day, count in daily_user_counts[-10:]:
        print(f"  第{day}天: {count:,}个用户")
    
    # 验证数据集结构
    print(f"\n数据集结构验证:")
    
    # 检查是否符合官方描述
    complete_users = set()  # 完整数据的用户
    masked_users = set()    # 有掩码的用户
    
    for uid in all_users:
        if uid in mask_users:
            masked_users.add(uid)
        else:
            complete_users.add(uid)
    
    print(f"完整数据用户数: {len(complete_users)}")
    print(f"有掩码数据用户数: {len(masked_users)}")
    
    if len(complete_users) > 0:
        print(f"完整数据用户ID范围: {min(complete_users)} - {max(complete_users)}")
    if len(masked_users) > 0:
        print(f"掩码数据用户ID范围: {min(masked_users)} - {max(masked_users)}")
    
    # 根据官方说明验证
    expected_complete = 80000
    expected_masked = 20000
    expected_total = 100000
    
    print(f"\n与官方说明对比:")
    print(f"官方说明 - 完整用户: {expected_complete}, 掩码用户: {expected_masked}, 总计: {expected_total}")
    print(f"实际发现 - 完整用户: {len(complete_users)}, 掩码用户: {len(masked_users)}, 总计: {len(all_users)}")
    
    return {
        'total_rows': total_rows,
        'total_users': len(all_users),
        'user_range': (min(all_users), max(all_users)),
        'complete_users': complete_users,
        'masked_users': masked_users,
        'mask_count': mask_count,
        'user_record_counts': user_record_counts,
        'daily_user_counts': dict(daily_user_counts)
    }

def quick_user_check(file_path, target_users=[50000, 80000, 90000, 99999]):
    """
    快速检查特定用户是否存在
    """
    print(f"\n快速检查特定用户是否存在:")
    print("-" * 40)
    
    found_users = set()
    chunk_size = 100000
    
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk_users = set(chunk['uid'].unique())
            found_target_users = chunk_users.intersection(target_users)
            
            if found_target_users:
                found_users.update(found_target_users)
                for uid in found_target_users:
                    print(f"✓ 找到用户 {uid}")
            
            # 如果找到所有目标用户就停止
            if len(found_users) >= len(target_users):
                break
    
    except Exception as e:
        print(f"检查用户时出错: {e}")
    
    missing_users = set(target_users) - found_users
    if missing_users:
        print(f"✗ 未找到用户: {sorted(missing_users)}")
    
    return found_users

if __name__ == "__main__":
    # file_path = "yjmob100k-dataset1.csv"
    file_path = "yjmob100k-dataset2.csv"
    
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
    else:
        print("开始暴力分析...")
        print("这会需要几分钟时间，请耐心等待...")
        
        # 先快速检查高ID用户
        quick_user_check(file_path)
        
        # 完整暴力分析
        results = brute_force_analysis(file_path)
        
        print("\n分析完成！")