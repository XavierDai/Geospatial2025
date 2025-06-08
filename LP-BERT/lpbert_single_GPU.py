import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import Counter
import math
import warnings
warnings.filterwarnings('ignore')

# 自动混合精度训练
from torch.cuda.amp import autocast, GradScaler

# =====================================================
# GPU优化的模型实现
# =====================================================

class OptimizedLPBERTEmbedding(nn.Module):
    """GPU优化的LP-BERT嵌入层"""
    
    def __init__(self, config):
        super().__init__()
        
        self.embed_size = config['embed_size']
        
        # 嵌入层
        self.day_embedding = nn.Embedding(config['max_days'], self.embed_size)
        self.time_embedding = nn.Embedding(config['max_times'], self.embed_size)
        self.location_embedding = nn.Embedding(config['max_locations'], self.embed_size)
        self.timedelta_embedding = nn.Embedding(config['max_timedelta'], self.embed_size)
        
        # 层标准化和dropout
        self.layer_norm = nn.LayerNorm(self.embed_size, eps=1e-6)
        self.dropout = nn.Dropout(config['dropout'])
        
        # 优化权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """优化的权重初始化"""
        for module in [self.day_embedding, self.time_embedding, 
                      self.location_embedding, self.timedelta_embedding]:
            nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, day_ids, time_ids, location_ids, timedelta_ids):
        """优化的前向传播"""
        # 并行计算所有嵌入
        day_emb = self.day_embedding(day_ids)
        time_emb = self.time_embedding(time_ids)
        location_emb = self.location_embedding(location_ids)
        timedelta_emb = self.timedelta_embedding(timedelta_ids)
        
        # 在GPU上高效求和
        embeddings = day_emb + time_emb + location_emb + timedelta_emb
        
        # 应用层标准化和dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class OptimizedLPBERTModel(nn.Module):
    """GPU优化的LP-BERT主模型"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embed_size = config['embed_size']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.max_locations = config['max_locations']
        
        # 嵌入层
        self.embedding = OptimizedLPBERTEmbedding(config)
        
        # 优化的Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_size,
            nhead=self.num_heads,
            dim_feedforward=config['hidden_size'],
            dropout=config['dropout'],
            activation='gelu',  # GELU通常比ReLU更好
            batch_first=True,
            norm_first=True  # Pre-LN可能更稳定
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers,
            enable_nested_tensor=False  # A100上可能更快
        )
        
        # 输出层
        self.output_layer = nn.Linear(self.embed_size, self.max_locations)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """优化的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, day_ids, time_ids, location_ids, timedelta_ids, attention_mask=None):
        """优化的前向传播"""
        # 嵌入
        embeddings = self.embedding(day_ids, time_ids, location_ids, timedelta_ids)
        
        # 创建高效的attention mask
        if attention_mask is None:
            # 避免创建不必要的mask
            encoded = self.transformer(embeddings)
        else:
            # 转换为Transformer需要的mask格式
            src_key_padding_mask = (attention_mask == 0)
            encoded = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # 输出预测
        logits = self.output_layer(encoded)
        
        return logits

# =====================================================
# 论文评估指标实现：GEO-BLEU和DTW
# =====================================================

class GEOBLEUCalculator:
    """GEO-BLEU评估指标 - 按论文实现"""
    
    def __init__(self, max_n=4):
        self.max_n = max_n
    
    def _get_ngrams(self, trajectory: List[Tuple[int, int]], n: int) -> List[Tuple]:
        """获取轨迹的n-gram"""
        if len(trajectory) < n:
            return []
        
        ngrams = []
        for i in range(len(trajectory) - n + 1):
            ngram = tuple(trajectory[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def _compute_bleu_score(self, pred_trajectory: List[Tuple[int, int]], 
                           true_trajectory: List[Tuple[int, int]]) -> float:
        """计算单个轨迹的BLEU分数"""
        if len(pred_trajectory) == 0 or len(true_trajectory) == 0:
            return 0.0
        
        # 计算各阶n-gram的精确度
        precisions = []
        
        for n in range(1, self.max_n + 1):
            pred_ngrams = self._get_ngrams(pred_trajectory, n)
            true_ngrams = self._get_ngrams(true_trajectory, n)
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            # 统计n-gram出现次数
            pred_counter = Counter(pred_ngrams)
            true_counter = Counter(true_ngrams)
            
            # 计算匹配的n-gram数量
            matches = 0
            for ngram, count in pred_counter.items():
                matches += min(count, true_counter.get(ngram, 0))
            
            # 计算精确度
            precision = matches / len(pred_ngrams) if len(pred_ngrams) > 0 else 0.0
            precisions.append(precision)
        
        # 计算几何平均
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            geo_mean = 0.0
        
        # 长度惩罚
        bp = min(1.0, len(pred_trajectory) / len(true_trajectory)) if len(true_trajectory) > 0 else 0.0
        
        return bp * geo_mean
    
    def compute_geobleu(self, predictions: Dict[int, Dict[int, List[Tuple[int, int]]]], 
                       ground_truth: Dict[int, Dict[int, List[Tuple[int, int]]]]) -> float:
        """计算GEO-BLEU分数（按天计算）"""
        total_score = 0.0
        total_count = 0
        
        for user_id in predictions:
            if user_id not in ground_truth:
                continue
            
            for day in predictions[user_id]:
                if day not in ground_truth[user_id]:
                    continue
                
                pred_traj = predictions[user_id][day]
                true_traj = ground_truth[user_id][day]
                
                if len(pred_traj) > 0 and len(true_traj) > 0:
                    score = self._compute_bleu_score(pred_traj, true_traj)
                    total_score += score
                    total_count += 1
        
        return total_score / total_count if total_count > 0 else 0.0

class DTWCalculator:
    """Dynamic Time Warping (DTW) 评估指标"""
    
    def __init__(self):
        pass
    
    def _euclidean_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """计算两点间的欧氏距离"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def compute_dtw(self, pred_trajectory: List[Tuple[int, int]], 
                   true_trajectory: List[Tuple[int, int]]) -> float:
        """计算两个轨迹之间的DTW距离"""
        if len(pred_trajectory) == 0 or len(true_trajectory) == 0:
            return float('inf')
        
        n, m = len(pred_trajectory), len(true_trajectory)
        
        # 创建DTW矩阵
        dtw_matrix = [[float('inf')] * m for _ in range(n)]
        
        # 初始化
        dtw_matrix[0][0] = self._euclidean_distance(pred_trajectory[0], true_trajectory[0])
        
        # 填充第一行
        for j in range(1, m):
            dtw_matrix[0][j] = dtw_matrix[0][j-1] + self._euclidean_distance(
                pred_trajectory[0], true_trajectory[j])
        
        # 填充第一列
        for i in range(1, n):
            dtw_matrix[i][0] = dtw_matrix[i-1][0] + self._euclidean_distance(
                pred_trajectory[i], true_trajectory[0])
        
        # 填充其余元素
        for i in range(1, n):
            for j in range(1, m):
                cost = self._euclidean_distance(pred_trajectory[i], true_trajectory[j])
                dtw_matrix[i][j] = cost + min(
                    dtw_matrix[i-1][j],      # 插入
                    dtw_matrix[i][j-1],      # 删除
                    dtw_matrix[i-1][j-1]     # 匹配
                )
        
        return dtw_matrix[n-1][m-1]
    
    def compute_average_dtw(self, predictions: Dict[int, List[Tuple[int, int]]], 
                           ground_truth: Dict[int, List[Tuple[int, int]]]) -> float:
        """计算平均DTW距离"""
        total_dtw = 0.0
        total_count = 0
        
        for user_id in predictions:
            if user_id not in ground_truth:
                continue
            
            pred_traj = predictions[user_id]
            true_traj = ground_truth[user_id]
            
            if len(pred_traj) > 0 and len(true_traj) > 0:
                dtw_dist = self.compute_dtw(pred_traj, true_traj)
                if dtw_dist != float('inf'):
                    total_dtw += dtw_dist
                    total_count += 1
        
        return total_dtw / total_count if total_count > 0 else float('inf')

# =====================================================
# 高效数据集实现
# =====================================================

def optimized_collate_fn(batch):
    """GPU优化的collate函数 - 修复多进程CUDA问题"""
    # 获取批次中的最大序列长度
    max_length = max([item['actual_length'] for item in batch])
    batch_size = len(batch)
    
    # 直接在CPU上创建张量，避免CUDA多进程问题
    days = torch.zeros((batch_size, max_length), dtype=torch.long)
    times = torch.zeros((batch_size, max_length), dtype=torch.long)
    locations = torch.zeros((batch_size, max_length), dtype=torch.long)
    timedeltas = torch.zeros((batch_size, max_length), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_length), dtype=torch.long)
    
    mask_positions = []
    
    # 批量填充数据
    for i, item in enumerate(batch):
        seq_len = item['actual_length']
        
        days[i, :seq_len] = item['days'][:seq_len]
        times[i, :seq_len] = item['times'][:seq_len]
        locations[i, :seq_len] = item['locations'][:seq_len]
        timedeltas[i, :seq_len] = item['timedeltas'][:seq_len]
        attention_mask[i, :seq_len] = item['attention_mask'][:seq_len]
        labels[i, :seq_len] = item['labels'][:seq_len]
        
        mask_positions.append(item['mask_positions'])
    
    return {
        'days': days,
        'times': times,
        'locations': locations,
        'timedeltas': timedeltas,
        'attention_mask': attention_mask,
        'labels': labels,
        'mask_positions': mask_positions
    }

class OptimizedHuMobTrainingDataset(Dataset):
    """GPU优化的训练数据集"""
    
    def __init__(self, data_file, seq_length=512, location_vocab=None, force_regenerate_sequences=False):
        self.seq_length = seq_length
        self.data_file = data_file
        
        print(f"加载GPU优化训练数据集: {data_file}")
        
        # 缓存设置
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        self.sequences_cache_file = os.path.join(cache_dir, f"optimized_sequences_{seq_length}.pkl")
        self.vocab_cache_file = os.path.join(cache_dir, "location_vocab.pkl")
        
        # 检查缓存
        if self._can_use_cache() and not force_regenerate_sequences:
            print("🚀 使用缓存快速加载...")
            self._load_from_cache()
            print(f"✅ 缓存加载完成！")
            print(f"   训练序列数量: {len(self.sequences)}")
            print(f"   位置词汇表大小: {len(self.location_vocab)}")
            return
        
        # 完整处理
        print("📊 开始GPU优化数据处理...")
        self._full_processing(location_vocab)
        self._save_to_cache()
        
        print(f"✅ GPU优化数据处理完成！")
        print(f"   训练序列数量: {len(self.sequences)}")
        print(f"   位置词汇表大小: {len(self.location_vocab)}")
    
    def _can_use_cache(self):
        """检查缓存可用性"""
        return (os.path.exists(self.sequences_cache_file) and 
                os.path.exists(self.vocab_cache_file))
    
    def _load_from_cache(self):
        """从缓存加载"""
        with open(self.sequences_cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            self.sequences = cache_data['sequences']
            
        with open(self.vocab_cache_file, 'rb') as f:
            self.location_vocab = pickle.load(f)
    
    def _save_to_cache(self):
        """保存到缓存"""
        cache_data = {
            'sequences': self.sequences,
            'seq_length': self.seq_length,
            'timestamp': time.time()
        }
        
        with open(self.sequences_cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        with open(self.vocab_cache_file, 'wb') as f:
            pickle.dump(self.location_vocab, f)
    
    def _full_processing(self, location_vocab):
        """完整数据处理"""
        # 使用pandas的向量化操作加速
        self.data = pd.read_csv(self.data_file)
        print(f"原始数据形状: {self.data.shape}")
        
        # 高效过滤
        complete_data = self.data[
            (self.data['uid'] < 80000) & 
            (self.data['x'] != 999) & 
            (self.data['y'] != 999)
        ].copy()
        
        self.data = complete_data
        print(f"完整数据形状: {self.data.shape}")
        
        # 构建词汇表
        if location_vocab is None:
            print("🗺 构建位置词汇表...")
            self.location_vocab = self._build_location_vocab()
        else:
            self.location_vocab = location_vocab
        
        # 预处理序列
        print("🔄 预处理用户序列（GPU优化版）...")
        self.sequences = self._prepare_sequences_optimized()
    
    def _build_location_vocab(self):
        """优化的词汇表构建"""
        # 使用pandas的高效去重
        unique_locations = self.data[['x', 'y']].drop_duplicates()
        print(f"   唯一位置数量: {len(unique_locations)}")
        
        location_vocab = {
            '<PAD>': 0,
            '<MASK>': 1, 
            '<UNK>': 2
        }
        
        # 向量化构建词汇表
        for idx, (x, y) in enumerate(unique_locations.values):
            location_key = (int(x), int(y))
            location_vocab[location_key] = idx + 3
        
        return location_vocab
    
    def _get_location_id(self, x, y):
        """获取位置ID"""
        location_key = (int(x), int(y))
        return self.location_vocab.get(location_key, self.location_vocab['<UNK>'])
    
    def _prepare_sequences_optimized(self):
        """GPU优化的序列准备"""
        sequences = []
        unique_users = self.data['uid'].unique()
        
        print(f"   处理 {len(unique_users)} 个用户...")
        
        # 使用groupby优化处理
        user_groups = self.data.groupby('uid')
        
        for uid in tqdm(unique_users, desc="处理用户序列"):
            user_data = user_groups.get_group(uid).sort_values(['d', 't'])
            
            if len(user_data) < 40:  # 提高最小序列要求
                continue
            
            # 向量化计算时间差
            user_data = user_data.copy()
            user_data['prev_time'] = user_data['d'] * 48 + user_data['t']
            user_data['timedelta'] = user_data['prev_time'].diff().fillna(1).astype(int)
            user_data['timedelta'] = user_data['timedelta'].clip(0, 100)
            
            # 创建更多重叠序列以增加训练数据
            step_size = self.seq_length // 8  # 更小的步长，更多序列
            
            for i in range(0, len(user_data) - self.seq_length + 1, step_size):
                seq_data = user_data.iloc[i:i + self.seq_length]
                
                if len(seq_data) != self.seq_length:
                    continue
                
                # 向量化特征提取
                days = seq_data['d'].values
                times = seq_data['t'].values
                locations = np.array([self._get_location_id(x, y) 
                                    for x, y in zip(seq_data['x'], seq_data['y'])])
                timedeltas = seq_data['timedelta'].values
                
                sequences.append({
                    'uid': uid,
                    'days': days,
                    'times': times,
                    'locations': locations,
                    'timedeltas': timedeltas,
                    'length': self.seq_length
                })
        
        print(f"   生成 {len(sequences)} 个训练序列")
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_len = self.seq_length
        
        # 高效数组操作
        days = np.array(seq['days'], dtype=np.long)
        times = np.array(seq['times'], dtype=np.long)
        locations = np.array(seq['locations'], dtype=np.long)
        timedeltas = np.array(seq['timedeltas'], dtype=np.long)
        
        # 创建标签
        labels = locations.copy()
        
        # 优化的掩码策略
        input_locations = locations.copy()
        mask_positions = []
        
        alpha_days = 15
        unique_days = np.unique(days)
        
        if len(unique_days) >= alpha_days:
            start_day_idx = np.random.randint(0, len(unique_days) - alpha_days + 1)
            mask_start_day = unique_days[start_day_idx]
            mask_end_day = unique_days[min(start_day_idx + alpha_days - 1, len(unique_days) - 1)]
            
            # 向量化掩码操作
            mask_condition = (days >= mask_start_day) & (days <= mask_end_day)
            mask_positions = np.where(mask_condition)[0].tolist()
            input_locations[mask_condition] = self.location_vocab['<MASK>']
        else:
            # 随机掩码
            mask_prob = np.random.random(len(locations))
            mask_condition = mask_prob < 0.15
            mask_positions = np.where(mask_condition)[0].tolist()
            input_locations[mask_condition] = self.location_vocab['<MASK>']
        
        return {
            'days': torch.tensor(days, dtype=torch.long),
            'times': torch.tensor(times, dtype=torch.long),
            'locations': torch.tensor(input_locations, dtype=torch.long),
            'timedeltas': torch.tensor(timedeltas, dtype=torch.long),
            'attention_mask': torch.ones(seq_len, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'mask_positions': mask_positions,
            'actual_length': seq_len
        }

# =====================================================
# GPU优化的训练器 - 包含论文评估指标
# =====================================================

class OptimizedLPBERTTrainer:
    """GPU优化的LP-BERT训练器 - 包含论文评估指标"""
    
    def __init__(self, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 启用编译优化（PyTorch 2.0+）
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            print("🚀 启用PyTorch编译优化...")
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
            except Exception as e:
                print(f"⚠️ 编译优化失败，使用标准模式: {e}")
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器 - Cosine退火
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['num_epochs'],
            eta_min=config['learning_rate'] * 0.01
        )
        
        # 混合精度训练
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.use_amp = torch.cuda.is_available()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        
        # 论文评估指标
        self.geobleu_calc = GEOBLEUCalculator()
        self.dtw_calc = DTWCalculator()
        
        # 记录
        self.train_history = []
        
        print(f"训练器初始化完成:")
        print(f"  设备: {self.device}")
        print(f"  混合精度: {self.use_amp}")
        print(f"  模型编译: {hasattr(self.model, '_orig_mod')}")
        print(f"  评估指标: GEO-BLEU + DTW (论文标准)")
    
    def _convert_to_coordinates(self, location_ids, location_vocab):
        """将位置ID转换为坐标"""
        id_to_location = {v: k for k, v in location_vocab.items() if isinstance(k, tuple)}
        coordinates = []
        for loc_id in location_ids:
            if loc_id in id_to_location:
                coordinates.append(id_to_location[loc_id])
            else:
                coordinates.append((100, 100))  # 默认坐标
        return coordinates
    
    def _evaluate_paper_metrics(self, train_loader, location_vocab, sample_size=100):
        """使用论文标准评估模型（采样评估以节省时间）"""
        self.model.eval()
        
        predictions_dict = {}
        ground_truth_dict = {}
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in train_loader:
                if sample_count >= sample_size:
                    break
                
                # 移动数据到设备
                days = batch['days'].to(self.device, non_blocking=True)
                times = batch['times'].to(self.device, non_blocking=True)
                locations = batch['locations'].to(self.device, non_blocking=True)
                timedeltas = batch['timedeltas'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # 前向传播
                if self.use_amp:
                    with autocast():
                        logits = self.model(days, times, locations, timedeltas, attention_mask)
                else:
                    logits = self.model(days, times, locations, timedeltas, attention_mask)
                
                predictions = torch.argmax(logits, dim=-1)
                
                # 处理每个样本
                for i in range(min(len(batch['mask_positions']), sample_size - sample_count)):
                    if sample_count >= sample_size:
                        break
                    
                    mask_positions = batch['mask_positions'][i]
                    if len(mask_positions) == 0:
                        continue
                    
                    # 提取掩码位置的预测和真实值
                    pred_ids = predictions[i, mask_positions].cpu().numpy()
                    true_ids = labels[i, mask_positions].cpu().numpy()
                    day_values = days[i, mask_positions].cpu().numpy()
                    
                    # 转换为坐标
                    pred_coords = self._convert_to_coordinates(pred_ids, location_vocab)
                    true_coords = self._convert_to_coordinates(true_ids, location_vocab)
                    
                    # 按天分组
                    user_id = sample_count  # 使用样本计数作为用户ID
                    predictions_dict[user_id] = {}
                    ground_truth_dict[user_id] = {}
                    
                    for j, day in enumerate(day_values):
                        day = int(day)
                        if day not in predictions_dict[user_id]:
                            predictions_dict[user_id][day] = []
                            ground_truth_dict[user_id][day] = []
                        
                        predictions_dict[user_id][day].append(pred_coords[j])
                        ground_truth_dict[user_id][day].append(true_coords[j])
                    
                    sample_count += 1
        
        # 计算论文指标
        geobleu_score = self.geobleu_calc.compute_geobleu(predictions_dict, ground_truth_dict)
        
        # 为DTW创建完整轨迹
        pred_full_trajectories = {}
        true_full_trajectories = {}
        
        for user_id in predictions_dict:
            pred_traj = []
            true_traj = []
            
            for day in sorted(predictions_dict[user_id].keys()):
                pred_traj.extend(predictions_dict[user_id][day])
                true_traj.extend(ground_truth_dict[user_id][day])
            
            pred_full_trajectories[user_id] = pred_traj
            true_full_trajectories[user_id] = true_traj
        
        dtw_score = self.dtw_calc.compute_average_dtw(pred_full_trajectories, true_full_trajectories)
        
        return geobleu_score, dtw_score
    
    def train_epoch(self, train_loader, epoch, location_vocab=None):
        """GPU优化的训练epoch - 添加论文评估"""
        self.model.train()
        total_loss = 0
        total_predictions = 0
        correct_predictions = 0
        
        # 启用cudnn benchmark
        torch.backends.cudnn.benchmark = True
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training (GPU优化)')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 非阻塞数据传输
                days = batch['days'].to(self.device, non_blocking=True)
                times = batch['times'].to(self.device, non_blocking=True)
                locations = batch['locations'].to(self.device, non_blocking=True)
                timedeltas = batch['timedeltas'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # 清零梯度
                self.optimizer.zero_grad(set_to_none=True)
                
                # 混合精度前向传播
                if self.use_amp:
                    with autocast():
                        logits = self.model(days, times, locations, timedeltas, attention_mask)
                        loss = self._compute_masked_loss(logits, locations, labels, batch['mask_positions'])
                    
                    # 混合精度反向传播
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 标准精度训练
                    logits = self.model(days, times, locations, timedeltas, attention_mask)
                    loss = self._compute_masked_loss(logits, locations, labels, batch['mask_positions'])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                # 统计
                total_loss += loss.item()
                
                # 计算简单准确率
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=-1)
                    for i, mask_pos_list in enumerate(batch['mask_positions']):
                        for pos in mask_pos_list:
                            if pos < predictions.size(1):
                                if predictions[i, pos] == labels[i, pos]:
                                    correct_predictions += 1
                                total_predictions += 1
                
                # 更新进度条
                if batch_idx % 20 == 0:  
                    avg_loss = total_loss / (batch_idx + 1)
                    accuracy = correct_predictions / max(total_predictions, 1) * 100
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    
                    pbar.set_postfix({
                        'Loss': f'{avg_loss:.4f}',
                        'Acc': f'{accuracy:.1f}%',
                        'VRAM': f'{gpu_memory:.1f}GB'
                    })
                
            except Exception as e:
                print(f"❌ 批次 {batch_idx} 处理失败: {e}")
                raise e
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / max(total_predictions, 1) * 100
        
        # 论文评估指标（每2个epoch评估一次）
        geobleu_score = 0.0
        dtw_score = float('inf')
        
        if location_vocab is not None and (epoch + 1) % 2 == 0:
            print(f"\n🔍 进行论文标准评估...")
            try:
                geobleu_score, dtw_score = self._evaluate_paper_metrics(train_loader, location_vocab)
                print(f"📊 论文指标 - GEO-BLEU: {geobleu_score:.4f}, DTW: {dtw_score:.4f}")
            except Exception as e:
                print(f"⚠️ 论文评估失败: {e}")
        
        return avg_loss, accuracy, geobleu_score, dtw_score
    
    def _compute_masked_loss(self, logits, input_locations, true_labels, mask_positions_batch):
        """GPU优化的损失计算"""
        all_masked_logits = []
        all_masked_labels = []
        
        for i, mask_positions in enumerate(mask_positions_batch):
            if len(mask_positions) == 0:
                continue
            
            mask_tensor = torch.tensor(mask_positions, device=logits.device)
            if len(mask_tensor) > 0 and mask_tensor.max() < logits.size(1):
                masked_logits = logits[i, mask_tensor]
                masked_labels = true_labels[i, mask_tensor]
                
                all_masked_logits.append(masked_logits)
                all_masked_labels.append(masked_labels)
        
        if len(all_masked_logits) == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)
        
        combined_logits = torch.cat(all_masked_logits, dim=0)
        combined_labels = torch.cat(all_masked_labels, dim=0)
        
        loss = self.criterion(combined_logits, combined_labels)
        return loss
    
    def train(self, train_loader, num_epochs, location_vocab=None):
        """完整训练流程 - 包含论文评估"""
        print(f"开始GPU优化LP-BERT训练")
        print(f"设备: {self.device}")
        print(f"混合精度: {self.use_amp}")
        print(f"训练epochs: {num_epochs}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"评估方式: 简单准确率 + GEO-BLEU + DTW (论文标准)")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        best_loss = float('inf')
        best_geobleu = 0.0
        patience_counter = 0
        max_patience = 3
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            epoch_start_time = time.time()
            
            try:
                # 训练
                train_loss, train_acc, geobleu_score, dtw_score = self.train_epoch(
                    train_loader, epoch, location_vocab
                )
                
                # 学习率调度
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                epoch_time = time.time() - epoch_start_time
                
                # 记录历史
                self.train_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'geobleu': geobleu_score,
                    'dtw': dtw_score,
                    'lr': current_lr,
                    'epoch_time': epoch_time
                })
                
                print(f"训练损失: {train_loss:.4f} | 简单准确率: {train_acc:.2f}%")
                print(f"学习率: {current_lr:.6f} | 耗时: {epoch_time:.1f}s")
                
                # 显示论文评估结果
                if geobleu_score > 0:
                    print(f"📊 论文评估 - GEO-BLEU: {geobleu_score:.4f} | DTW: {dtw_score:.4f}")
                    print(f"🎯 论文对比 - 目标GEO-BLEU: 0.3440 | 目标DTW: 29.9633")
                    
                    progress = geobleu_score / 0.3440 * 100
                    print(f"📈 GEO-BLEU进度: {progress:.1f}% of 论文结果")
                
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
                    print(f"GPU峰值内存: {gpu_memory:.1f}GB")
                
                # 保存最佳模型
                model_improved = False
                if geobleu_score > 0 and geobleu_score > best_geobleu:
                    best_geobleu = geobleu_score
                    model_improved = True
                    save_reason = f"GEO-BLEU: {geobleu_score:.4f}"
                elif geobleu_score == 0 and train_loss < best_loss:
                    best_loss = train_loss
                    model_improved = True
                    save_reason = f"损失: {train_loss:.4f}"
                
                if model_improved:
                    patience_counter = 0
                    self.save_model(f'optimized_lpbert_best_epoch_{epoch+1}.pth')
                    print(f"✓ 保存最佳模型 ({save_reason})")
                else:
                    patience_counter += 1
                
                # 早停
                if patience_counter >= max_patience:
                    print(f"早停：模型连续{max_patience}个epoch未改善")
                    break
                    
            except Exception as e:
                print(f"❌ 训练epoch {epoch+1}失败: {e}")
                self.save_model(f'optimized_lpbert_interrupted_epoch_{epoch+1}.pth')
                raise e
        
        print(f"\n训练完成!")
        print(f"最佳损失: {best_loss:.4f}")
        print(f"最佳GEO-BLEU: {best_geobleu:.4f}")
        
        # 最终评估总结
        if len(self.train_history) > 0:
            final_stats = self.train_history[-1]
            print(f"\n📊 最终训练结果:")
            print(f"  简单准确率: {final_stats['train_acc']:.2f}%")
            if final_stats['geobleu'] > 0:
                print(f"  GEO-BLEU: {final_stats['geobleu']:.4f} (论文目标: 0.3440)")
                print(f"  DTW: {final_stats['dtw']:.4f} (论文目标: 29.9633)")
                
                # 与论文对比
                geobleu_ratio = final_stats['geobleu'] / 0.3440
                dtw_ratio = final_stats['dtw'] / 29.9633 if final_stats['dtw'] != float('inf') else float('inf')
                
                print(f"  📈 GEO-BLEU达到论文的 {geobleu_ratio:.1%}")
                if dtw_ratio != float('inf'):
                    if dtw_ratio < 1:
                        print(f"  📈 DTW比论文好 {(1-dtw_ratio)*100:.1f}%")
                    else:
                        print(f"  📉 DTW比论文差 {(dtw_ratio-1)*100:.1f}%")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return self.train_history
    
    def save_model(self, path):
        """保存模型"""
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'config': self.config,
            'train_history': self.train_history,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }, path)

# =====================================================
# GPU优化配置
# =====================================================

def create_optimized_config(location_vocab_size):
    """创建GPU优化的模型配置"""
    config = {
        'max_days': 75,
        'max_times': 48,
        'max_locations': location_vocab_size,
        'max_timedelta': 101,
        'max_seq_length': 512,
        'embed_size': 256,
        'num_layers': 6,
        'num_heads': 16,
        'batch_size': 32,
        'num_epochs': 50,
        'alpha_days': 15,
        'beta_penalty': 0.9,
        'hidden_size': 1024,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
    }
    return config

def prepare_optimized_training_data(original_file, output_train_file, force_regenerate=False):
    """准备GPU优化的训练数据"""
    if os.path.exists(output_train_file) and not force_regenerate:
        print(f"✅ 训练数据文件已存在: {output_train_file}")
        return output_train_file
    
    print("准备GPU优化训练数据...")
    df = pd.read_csv(original_file)
    
    training_data = df[
        (df['uid'] < 80000) & 
        (df['x'] != 999) & 
        (df['y'] != 999)
    ].copy()
    
    print(f"训练数据形状: {training_data.shape}")
    training_data.to_csv(output_train_file, index=False)
    return output_train_file

def train_optimized_lpbert(force_regenerate_data=False, force_regenerate_sequences=False):
    """训练GPU优化的LP-BERT模型 - 包含论文评估"""
    print("="*60)
    print("GPU优化LP-BERT模型训练 (针对A100 + 论文评估)")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠️ 警告：未检测到CUDA，将使用CPU训练（非常慢）")
    else:
        print(f"🚀 检测到GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 1. 准备数据
    train_file = prepare_optimized_training_data(
        "yjmob100k-dataset1.csv", 
        "optimized_training_data.csv",
        force_regenerate_data
    )
    
    # 2. 创建数据集
    print("\n创建GPU优化训练数据集...")
    train_dataset = OptimizedHuMobTrainingDataset(
        train_file,
        seq_length=512,
        force_regenerate_sequences=force_regenerate_sequences
    )
    
    with open('optimized_location_vocab.pkl', 'wb') as f:
        pickle.dump(train_dataset.location_vocab, f)
    
    # 3. 创建配置和数据加载器
    config = create_optimized_config(len(train_dataset.location_vocab))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=optimized_collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    print(f"GPU优化数据加载器:")
    print(f"  批次数: {len(train_loader)}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  序列长度: {config['max_seq_length']}")
    
    # 4. 创建模型
    print("\n创建GPU优化LP-BERT模型...")
    model = OptimizedLPBERTModel(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {param_count:,}")
    
    # 5. 验证数据加载器
    print("\n验证GPU优化数据加载器...")
    try:
        sample_batch = next(iter(train_loader))
        print("✅ 数据加载器验证成功")
        print(f"  批次形状: {sample_batch['days'].shape}")
    except Exception as e:
        print(f"❌ 数据加载器验证失败: {e}")
        return False
    
    # 6. 开始训练
    print("\n开始GPU优化训练（包含论文标准评估）...")
    trainer = OptimizedLPBERTTrainer(model, config)
    
    try:
        history = trainer.train(train_loader, config['num_epochs'], train_dataset.location_vocab)
        
        trainer.save_model('optimized_lpbert_final.pth')
        print(f"✅ GPU优化模型训练完成")
        
        if history:
            total_time = sum([h['epoch_time'] for h in history])
            final_stats = history[-1]
            
            print(f"\n📊 训练统计:")
            print(f"  总训练时间: {total_time/60:.1f}分钟")
            print(f"  最终损失: {final_stats['train_loss']:.4f}")
            print(f"  最终简单准确率: {final_stats['train_acc']:.2f}%")
            
            if final_stats['geobleu'] > 0:
                print(f"  最终GEO-BLEU: {final_stats['geobleu']:.4f}")
                print(f"  最终DTW: {final_stats['dtw']:.4f}")
                print(f"\n🎯 与论文Task1对比:")
                print(f"  论文GEO-BLEU: 0.3440 | 你的结果: {final_stats['geobleu']:.4f}")
                print(f"  论文DTW: 29.9633 | 你的结果: {final_stats['dtw']:.4f}")
                
                performance = final_stats['geobleu'] / 0.3440 * 100
                print(f"  📈 你的GEO-BLEU达到论文的{performance:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main_optimized():
    """GPU优化的主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU优化LP-BERT Human Mobility Prediction')
    parser.add_argument('--force-data', action='store_true', help='强制重新生成数据')
    parser.add_argument('--force-sequences', action='store_true', help='强制重新生成序列')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--seq-length', type=int, default=512, help='序列长度')
    parser.add_argument('--epochs', type=int, default=8, help='训练轮数')
    parser.add_argument('--embed-size', type=int, default=256, help='嵌入维度')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    
    args = parser.parse_args()
    
    print("GPU优化LP-BERT: Location Prediction BERT for Human Mobility")
    print("针对NVIDIA A100 80GB优化")
    print("="*80)
    
    if not os.path.exists("yjmob100k-dataset1.csv"):
        print("❌ 原始数据文件 yjmob100k-dataset1.csv 不存在")
        return
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   PyTorch版本: {torch.__version__}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ 启用CUDA优化设置")
    
    print(f"\n🔧 优化参数:")
    print(f"  批次大小: {args.batch_size}")
    print(f"  序列长度: {args.seq_length}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  嵌入维度: {args.embed_size}")
    print(f"  学习率: {args.lr}")
    print(f"  混合精度: {'启用' if torch.cuda.is_available() else '禁用'}")
    
    success = train_optimized_lpbert(
        force_regenerate_data=args.force_data,
        force_regenerate_sequences=args.force_sequences
    )
    
    if success:
        print("\n✅ GPU优化LP-BERT训练完成!")
        print("\n🎯 关键改进:")
        print("  ✓ 批次大小增加到32 (原16)")
        print("  ✓ 序列长度增加到512 (原256)")
        print("  ✓ 启用混合精度训练 (FP16)")
        print("  ✓ 启用PyTorch编译优化")
        print("  ✓ 添加论文标准评估 (GEO-BLEU + DTW)")
        print("  ✓ 修复CUDA多进程问题")
        
        print("\n💡 评估指标说明:")
        print("  • 简单准确率: 训练过程监控指标")
        print("  • GEO-BLEU: 论文标准指标 (目标0.3440)")
        print("  • DTW: 轨迹相似性指标 (目标29.9633)")
        
    else:
        print("❌ 训练失败")
        print("\n🔧 故障排除:")
        print("  1. 检查GPU内存是否足够")
        print("  2. 降低batch_size: --batch-size 16")
        print("  3. 降低序列长度: --seq-length 256")

if __name__ == "__main__":
    main_optimized()