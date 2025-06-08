import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import os
import pickle
import math
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


class LPBERTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 嵌入层
        self.day_embedding = nn.Embedding(config['max_days'], config['embed_size'])
        self.time_embedding = nn.Embedding(config['max_times'], config['embed_size'])
        self.location_embedding = nn.Embedding(config['max_locations'], config['embed_size'])
        self.timedelta_embedding = nn.Embedding(config['max_timedelta'], config['embed_size'])
        
        # 位置编码
        self.position_embedding = nn.Embedding(config['seq_length'], config['embed_size'])
        
        # 嵌入缩放因子
        self.embed_scale = math.sqrt(config['embed_size'])
        
        # 标准化和dropout
        self.layer_norm = nn.LayerNorm(config['embed_size'], eps=1e-12)
        self.dropout = nn.Dropout(config['dropout'])
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """针对大词汇表的初始化策略"""
        # 截断正态分布
        std = 0.02
        
        # 对location embedding使用更小的初始化
        nn.init.trunc_normal_(self.location_embedding.weight, mean=0, std=std/2, a=-2*std, b=2*std)
        
        # 其他embedding正常初始化
        for emb in [self.day_embedding, self.time_embedding, self.timedelta_embedding, self.position_embedding]:
            nn.init.trunc_normal_(emb.weight, mean=0, std=std, a=-2*std, b=2*std)
        
        # 特殊token初始化
        with torch.no_grad():
            self.location_embedding.weight[0].fill_(0)  # <PAD>
            self.location_embedding.weight[1].normal_(0, std)  # <MASK>
            if 2 < len(self.location_embedding.weight):
                self.location_embedding.weight[2].normal_(0, std)  # <UNK>
    
    def forward(self, day_ids, time_ids, location_ids, timedelta_ids):
        batch_size, seq_len = day_ids.shape
        
        # 位置索引
        position_ids = torch.arange(seq_len, dtype=torch.long, device=day_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 嵌入层（location embedding使用缩放）
        location_embeds = self.location_embedding(location_ids) * self.embed_scale
        
        embeddings = (
            self.day_embedding(day_ids) + 
            self.time_embedding(time_ids) + 
            location_embeds + 
            self.timedelta_embedding(timedelta_ids) +
            self.position_embedding(position_ids)
        )
        
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)

class LPBERTModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 嵌入层
        self.embedding = LPBERTEmbedding(config)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['embed_size'],
            nhead=config['num_heads'],
            dim_feedforward=config['hidden_size'],
            dropout=config['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config['num_layers']
        )
        
        # 输出层（两阶段投影）
        self.dense = nn.Linear(config['embed_size'], config['embed_size'])
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config['embed_size'], eps=1e-12)
        
        # 最终预测层（使用与embedding层共享的权重）
        self.decoder = nn.Linear(config['embed_size'], config['max_locations'], bias=False)
        
        # 权重共享
        self.decoder.weight = self.embedding.location_embedding.weight
        
        # 输出偏置
        self.bias = nn.Parameter(torch.zeros(config['max_locations']))
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        # Dense层初始化
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        
        # LayerNorm初始化
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)
        
        # 偏置初始化
        nn.init.zeros_(self.bias)
    
    def forward(self, day_ids, time_ids, location_ids, timedelta_ids, attention_mask=None):
        # 嵌入
        embeddings = self.embedding(day_ids, time_ids, location_ids, timedelta_ids)
        
        # 注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones_like(day_ids, dtype=torch.float32)
        
        # Transformer要求的padding mask
        src_key_padding_mask = (attention_mask == 0)
        
        # Transformer编码
        encoded = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # 输出变换
        encoded = self.dense(encoded)
        encoded = self.activation(encoded)
        encoded = self.layer_norm(encoded)
        
        # 预测
        logits = self.decoder(encoded) + self.bias
        
        return logits

class TrainingDataset(Dataset):
    """训练数据集 - 优化版"""
    
    def __init__(self, seq_length=256, mask_prob=0.15, mask_consecutive_prob=0.8):
        self.seq_length = seq_length
        self.mask_prob = mask_prob
        self.mask_consecutive_prob = mask_consecutive_prob
        
        print(f"📁 加载数据集...")
        
        # 加载词汇表
        with open("cache/location_vocab.pkl", 'rb') as f:
            self.location_vocab = pickle.load(f)
        
        # 加载序列
        with open("cache/training_sequences_256.pkl", 'rb') as f:
            cache_data = pickle.load(f)
            if isinstance(cache_data, dict) and 'sequences' in cache_data:
                sequences = cache_data['sequences']
            else:
                sequences = cache_data
        
        self.sequences = sequences
        print(f"✅ 加载完成: {len(self.sequences)} 条序列")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # 获取数据
        days = np.array(seq['days'], dtype=np.int64)
        times = np.array(seq['times'], dtype=np.int64)
        locations = np.array(seq['locations'], dtype=np.int64)
        timedeltas = np.array(seq['timedeltas'], dtype=np.int64)
        
        # 序列长度处理
        original_len = len(days)
        
        if original_len >= self.seq_length:
            # 随机截取
            start_idx = np.random.randint(0, original_len - self.seq_length + 1)
            end_idx = start_idx + self.seq_length
            
            days = days[start_idx:end_idx]
            times = times[start_idx:end_idx]
            locations = locations[start_idx:end_idx]
            timedeltas = timedeltas[start_idx:end_idx]
            actual_len = self.seq_length
        else:
            # 填充
            pad_len = self.seq_length - original_len
            days = np.pad(days, (0, pad_len), 'constant', constant_values=0)
            times = np.pad(times, (0, pad_len), 'constant', constant_values=0) 
            locations = np.pad(locations, (0, pad_len), 'constant', constant_values=0)
            timedeltas = np.pad(timedeltas, (0, pad_len), 'constant', constant_values=1)
            actual_len = original_len
        
        # 掩码处理
        labels = locations.copy()
        input_locations = locations.copy()
        mask_positions = []
        
        # 连续掩码 vs 随机掩码
        if np.random.random() < self.mask_consecutive_prob and actual_len > 10:
            # 连续掩码
            mask_len = np.random.randint(5, min(20, actual_len // 3))
            mask_start = np.random.randint(0, actual_len - mask_len)
            
            for i in range(mask_start, mask_start + mask_len):
                if locations[i] > 0:  # 不掩码PAD
                    mask_positions.append(i)
                    input_locations[i] = 1  # <MASK>
        else:
            # 随机掩码
            num_masks = max(1, int(actual_len * self.mask_prob))
            valid_positions = [i for i in range(actual_len) if locations[i] > 0]
            
            if valid_positions:
                mask_positions = np.random.choice(
                    valid_positions, 
                    size=min(num_masks, len(valid_positions)), 
                    replace=False
                ).tolist()
                
                for pos in mask_positions:
                    input_locations[pos] = 1  # <MASK>
        
        return {
            'days': torch.tensor(days, dtype=torch.long),
            'times': torch.tensor(times, dtype=torch.long),
            'locations': torch.tensor(input_locations, dtype=torch.long),
            'timedeltas': torch.tensor(timedeltas, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'mask_positions': mask_positions,
            'actual_length': actual_len
        }

def collate_fn(batch):
    """优化的批处理函数"""
    # 找到最大长度
    max_length = max(item['actual_length'] for item in batch)
    batch_size = len(batch)
    
    # 预分配张量
    days = torch.zeros((batch_size, max_length), dtype=torch.long)
    times = torch.zeros((batch_size, max_length), dtype=torch.long)
    locations = torch.zeros((batch_size, max_length), dtype=torch.long)
    timedeltas = torch.zeros((batch_size, max_length), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_length), dtype=torch.long)
    
    mask_positions = []
    
    for i, item in enumerate(batch):
        length = item['actual_length']
        days[i, :length] = item['days'][:length]
        times[i, :length] = item['times'][:length]
        locations[i, :length] = item['locations'][:length]
        timedeltas[i, :length] = item['timedeltas'][:length]
        attention_mask[i, :length] = 1.0
        labels[i, :length] = item['labels'][:length]
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

# =====================================================
# DDP训练
# =====================================================

def ddp_train_worker(rank, world_size, config):
    """DDP训练进程"""
    
    # 初始化
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print(f"🚀 启动DDP训练 ({world_size} GPUs)")
        print(f"配置: {config}")
    
    try:
        # 数据集
        dataset = TrainingDataset(
            seq_length=config['seq_length'],
            mask_prob=config['mask_prob']
        )
        
        # 采样器
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        
        # 数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        # 模型
        model = LPBERTModel(config).cuda()
        
        # 广播初始参数确保一致性
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        
        model = DDP(model, device_ids=[rank])
        
        # 优化器（使用更高的学习率）
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        num_training_steps = len(dataloader) * config['num_epochs']
        num_warmup_steps = int(config['warmup_ratio'] * num_training_steps)
        
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # 损失函数
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=config['label_smoothing'])
        
        # 混合精度
        scaler = GradScaler() if AMP_AVAILABLE else None
        
        # 训练循环
        best_accuracy = 0
        
        for epoch in range(config['num_epochs']):
            sampler.set_epoch(epoch)
            model.train()
            
            total_loss = 0
            correct = 0
            total = 0
            
            # 进度条
            if rank == 0:
                pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
            else:
                pbar = dataloader
            
            for step, batch in enumerate(pbar):
                # 数据到GPU
                days = batch['days'].cuda(non_blocking=True)
                times = batch['times'].cuda(non_blocking=True)
                locations = batch['locations'].cuda(non_blocking=True)
                timedeltas = batch['timedeltas'].cuda(non_blocking=True)
                attention_mask = batch['attention_mask'].cuda(non_blocking=True)
                labels = batch['labels'].cuda(non_blocking=True)
                
                # 前向传播
                if AMP_AVAILABLE and scaler is not None:
                    with autocast():
                        logits = model(days, times, locations, timedeltas, attention_mask)
                        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                    # 反向传播
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(days, times, locations, timedeltas, attention_mask)
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                
                # 统计
                total_loss += loss.item()
                
                # 计算准确率（仅掩码位置）
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=-1)
                    for i, mask_pos in enumerate(batch['mask_positions']):
                        for pos in mask_pos:
                            if pos < predictions.size(1) and pos < labels.size(1):
                                if predictions[i, pos] == labels[i, pos]:
                                    correct += 1
                                total += 1
                
                # 更新进度条
                if rank == 0 and step % 20 == 0:
                    acc = correct / max(total, 1) * 100
                    current_lr = scheduler.get_last_lr()[0]
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{acc:.2f}%',
                        'lr': f'{current_lr:.2e}'
                    })
            
            # Epoch结束
            if rank == 0:
                epoch_loss = total_loss / len(dataloader)
                epoch_acc = correct / max(total, 1) * 100
                
                print(f"\nEpoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
                
                # 保存最佳模型
                if epoch_acc > best_accuracy:
                    best_accuracy = epoch_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'accuracy': epoch_acc,
                        'config': config
                    }, 'lpbert_best_optimized.pth')
                    print(f"💾 保存最佳模型 (Acc: {epoch_acc:.2f}%)")
                
                # 定期保存
                if (epoch + 1) % 5 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'config': config
                    }, f'lpbert_epoch_{epoch+1}.pth')
        
        if rank == 0:
            print(f"\n✅ 训练完成! 最佳准确率: {best_accuracy:.2f}%")
    
    except Exception as e:
        print(f"❌ GPU {rank} 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        dist.destroy_process_group()

# =====================================================
# 主函数
# =====================================================

def create_optimized_config(vocab_size):
    """优化的配置"""
    return {
        # 数据参数
        'max_days': 75,
        'max_times': 48,
        'max_locations': vocab_size,
        'max_timedelta': 101,
        'seq_length': 256,
        
        # 模型参数
        'embed_size': 256,
        'num_layers': 6,
        'num_heads': 8,
        'hidden_size': 1024,
        'dropout': 0.1,
        
        # 训练参数
        'batch_size': 32,              # 增大batch size
        'num_epochs': 50,
        'learning_rate': 5e-4,         # 更高的学习率
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,           # 10% warmup
        'label_smoothing': 0.1,
        'max_grad_norm': 1.0,
        
        # 掩码参数
        'mask_prob': 0.15,
    }

def main():
    """主函数"""
    print("🚀 LP-BERT 优化训练")
    print("=" * 60)
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ 需要GPU")
        return
    
    world_size = torch.cuda.device_count()
    print(f"📊 检测到 {world_size} 个GPU")
    
    # 加载词汇表获取大小
    with open("cache/location_vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    
    # 创建配置
    config = create_optimized_config(len(vocab))
    
    print(f"\n⚙️ 配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # 启动训练
    print(f"\n🚀 启动DDP训练...")
    mp.spawn(ddp_train_worker, args=(world_size, config), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()