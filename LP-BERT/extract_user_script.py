#!/usr/bin/env python3
"""
简单脚本：提取特定用户的所有记录到新CSV文件
"""

import pandas as pd
import os
import time

def extract_user_data(input_file, user_id, output_file=None):
    """
    提取指定用户的所有记录到新CSV文件
    
    Args:
        input_file: 输入CSV文件路径
        user_id: 要提取的用户ID
        output_file: 输出CSV文件路径（可选，默认为user_{user_id}_data.csv）
    """
    
    if output_file is None:
        output_file = f"user_{user_id}_data.csv"
    
    print(f"开始提取用户 {user_id} 的数据...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return
    
    start_time = time.time()
    
    user_data_chunks = []
    total_records = 0
    chunks_processed = 0
    
    chunk_size = 100000  # 每次读取10万行
    
    try:
        print("开始扫描文件...")
        
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            chunks_processed += 1
            
            # 筛选指定用户的数据
            user_chunk = chunk[chunk['uid'] == user_id]
            
            if len(user_chunk) > 0:
                user_data_chunks.append(user_chunk)
                total_records += len(user_chunk)
                print(f"  Chunk {chunks_processed}: 找到 {len(user_chunk)} 条记录 (累计: {total_records})")
            
            # 每处理1000个chunk显示一次进度
            if chunks_processed % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  已处理 {chunks_processed} 个chunk, 用时 {elapsed:.1f}秒")
    
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 合并所有数据
    if user_data_chunks:
        print(f"\n合并数据...")
        user_df = pd.concat(user_data_chunks, ignore_index=True)
        
        # 按时间排序
        user_df = user_df.sort_values(['d', 't']).reset_index(drop=True)
        
        # 保存到新文件
        print(f"保存到文件: {output_file}")
        user_df.to_csv(output_file, index=False)
        
        # 显示统计信息
        elapsed = time.time() - start_time
        print(f"\n✅ 提取完成!")
        print(f"总用时: {elapsed:.1f}秒")
        print(f"提取的记录数: {len(user_df)}")
        print(f"时间跨度: 第{user_df['d'].min()}天 - 第{user_df['d'].max()}天")
        print(f"位置范围: X({user_df['x'].min()}-{user_df['x'].max()}), Y({user_df['y'].min()}-{user_df['y'].max()})")
        print(f"访问的唯一位置数: {len(user_df[['x', 'y']].drop_duplicates())}")
        
        # 显示前几行数据预览
        print(f"\n前10行数据预览:")
        print(user_df.head(10).to_string(index=False))
        
        return user_df
    
    else:
        print(f"\n❌ 未找到用户 {user_id} 的任何数据")
        return None

def extract_multiple_users(input_file, user_ids, output_dir="user_data"):
    """
    提取多个用户的数据
    
    Args:
        input_file: 输入CSV文件路径
        user_ids: 用户ID列表
        output_dir: 输出目录
    """
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    print(f"开始提取 {len(user_ids)} 个用户的数据...")
    
    for i, user_id in enumerate(user_ids):
        print(f"\n[{i+1}/{len(user_ids)}] 处理用户 {user_id}")
        
        output_file = os.path.join(output_dir, f"user_{user_id}_data.csv")
        extract_user_data(input_file, user_id, output_file)

def quick_user_preview(input_file, user_id, num_records=20):
    """
    快速预览用户数据（不保存文件）
    
    Args:
        input_file: 输入CSV文件路径
        user_id: 用户ID
        num_records: 显示的记录数
    """
    
    print(f"快速预览用户 {user_id} 的前 {num_records} 条记录...")
    
    chunk_size = 100000
    found_records = []
    
    try:
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            user_chunk = chunk[chunk['uid'] == user_id]
            
            if len(user_chunk) > 0:
                found_records.append(user_chunk)
                
                # 如果找到足够的记录就停止
                total_found = sum(len(df) for df in found_records)
                if total_found >= num_records:
                    break
        
        if found_records:
            user_df = pd.concat(found_records, ignore_index=True)
            user_df = user_df.sort_values(['d', 't']).reset_index(drop=True)
            
            print(f"找到 {len(user_df)} 条记录，显示前 {min(num_records, len(user_df))} 条:")
            print(user_df.head(num_records).to_string(index=False))
        else:
            print(f"未找到用户 {user_id} 的数据")
    
    except Exception as e:
        print(f"预览时出错: {e}")

if __name__ == "__main__":
    # 配置
    input_file = "yjmob100k-dataset1.csv"
    
    print("HuMob用户数据提取工具")
    print("=" * 50)
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        print("请修改脚本中的 input_file 变量为正确的文件路径")
        exit(1)
    
    # 提取用户0的数据
    print("正在提取用户0的数据...")
    user_0_data = extract_user_data(input_file, user_id=0, output_file="user_0_trajectory.csv")
    
    # 可选：提取其他用户
    print("\n" + "=" * 50)
    print("可选操作示例:")
    
    # 1. 快速预览其他用户
    print("\n1. 快速预览用户1的数据:")
    quick_user_preview(input_file, user_id=1, num_records=10)
    
    # 2. 提取多个用户（可选，注释掉以避免大量文件）
    # print("\n2. 提取多个用户数据:")
    # sample_users = [0, 1, 2, 3, 4]  # 提取前5个用户
    # extract_multiple_users(input_file, sample_users, output_dir="sample_users")
    
    print(f"\n✅ 脚本执行完成!")
    print(f"用户0的数据已保存到: user_0_trajectory.csv")
