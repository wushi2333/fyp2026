import os
import torch
import numpy as np
import bisect
from torch.utils.data import Dataset

class UniversalEEGDataset(Dataset):
    def __init__(self, root_dir, mode='train', augment=False, target_dataset='BCIC', 
                 snr_aug=False, snr_prob=0.5, num_segments=4):
        self.mode = mode
        self.augment = augment
        
        # S&R 参数
        self.snr_aug = snr_aug and (mode == 'train')
        self.snr_prob = snr_prob
        self.num_segments = num_segments
        
        # === 内存优化存储结构 ===
        # 不再使用 self.X = np.concatenate(...)
        # 而是保留碎片化的列表，避免内存峰值
        self.X_chunks = [] 
        self.y_chunks = []
        self.chunk_sizes = []
        self.cumulative_sizes = []
        
        target_file = 'train_data.npy' if mode == 'train' else 'test_data.npy'
        print(f"[{mode.upper()}] 正在扫描数据 (目标: {target_dataset})...")
        print(f" -> 提示: 启用了内存优化模式 (List-based storage)")
        
        total_samples = 0
        loaded_files = 0
        
        for root, dirs, files in os.walk(root_dir):
            if target_dataset != 'All' and target_dataset not in root:
                continue
                
            if target_file in files:
                file_path = os.path.join(root, target_file)
                try:
                    # 加载数据
                    data = np.load(file_path, allow_pickle=True).item()
                    X_part = data['X']
                    y_part = data['y']
                    
                    # === 关键优化 1: 立即转为 float32 节省一半内存 ===
                    if X_part.dtype != np.float32:
                        X_part = X_part.astype(np.float32)
                        
                    # === 关键优化 2: 存储碎片，不合并 ===
                    self.X_chunks.append(X_part)
                    self.y_chunks.append(torch.from_numpy(y_part).long())
                    
                    count = len(X_part)
                    self.chunk_sizes.append(count)
                    total_samples += count
                    loaded_files += 1
                    
                    # 打印进度防止以为卡死
                    if loaded_files % 10 == 0:
                        print(f"    ...已加载 {loaded_files} 个文件 ({total_samples} 样本)")
                        
                except Exception as e:
                    print(f"❌ 加载失败 {file_path}: {e}")

        # 构建累积索引，用于快速定位
        self.cumulative_sizes = np.cumsum(self.chunk_sizes)
        self.total_len = total_samples
        
        if total_samples > 0:
            print(f" -> ✅ 加载完成: {loaded_files} 文件, 共 {total_samples} 样本")
            
            # === S&R 索引构建 (优化版) ===
            if self.snr_aug:
                print(" -> 构建 S&R 全局索引 (这可能需要几秒钟)...")
                # 为了支持跨 chunk 的 S&R，我们需要一个全局的 label map
                # 这会消耗一些内存，但对于 S&R 是必须的
                self.global_labels = torch.cat(self.y_chunks) # 这个比较小，可以合并
                self.class_indices = {}
                unique_labels = torch.unique(self.global_labels).numpy()
                for label in unique_labels:
                    self.class_indices[label] = (self.global_labels == label).nonzero(as_tuple=True)[0]
                print(" -> S&R 索引构建完成")
        else:
            print(f"❌ 警告: 未找到任何数据!")

    def __len__(self):
        return self.total_len

    def _get_chunk_index(self, global_idx):
        """
        根据全局索引找到 (Chunk索引, Chunk内偏移量)
        使用二分查找加速
        """
        chunk_idx = bisect.bisect_right(self.cumulative_sizes, global_idx)
        if chunk_idx == 0:
            local_idx = global_idx
        else:
            local_idx = global_idx - self.cumulative_sizes[chunk_idx - 1]
        return chunk_idx, local_idx

    def _get_data_at(self, global_idx):
        chunk_idx, local_idx = self._get_chunk_index(global_idx)
        # 此时才将 Numpy 转为 Tensor，减少常驻内存开销
        x = torch.from_numpy(self.X_chunks[chunk_idx][local_idx])
        y = self.y_chunks[chunk_idx][local_idx]
        return x, y

    def _apply_snr(self, x_current, y_current):
        """
        优化版 S&R: 支持跨 Chunk 抽取
        """
        label = y_current.item()
        candidates = self.class_indices[label]
        
        n_bands, n_time, n_csp = x_current.shape
        segment_len = n_time // self.num_segments
        x_new = torch.zeros_like(x_current)
        
        for i in range(self.num_segments):
            start = i * segment_len
            end = start + segment_len if i < self.num_segments - 1 else n_time
            
            # 随机选择一个全局索引
            random_global_idx = candidates[torch.randint(0, len(candidates), (1,)).item()].item()
            
            # 获取源样本 (需要跨 Chunk 查找)
            chunk_idx, local_idx = self._get_chunk_index(random_global_idx)
            x_source_np = self.X_chunks[chunk_idx][local_idx]
            # 临时转 Tensor
            x_source = torch.from_numpy(x_source_np)
            
            x_new[:, start:end, :] = x_source[:, start:end, :]
            
        return x_new

    def __getitem__(self, idx):
        x_sample, y_sample = self._get_data_at(idx)
        
        # 1. S&R Augmentation
        if self.snr_aug:
            if torch.rand(1).item() < self.snr_prob:
                x_sample = self._apply_snr(x_sample, y_sample)

        # 2. Standard Augmentation
        if self.augment:
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(x_sample) * 0.05
                x_sample = x_sample + noise
            if torch.rand(1).item() > 0.5:
                scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
                x_sample = x_sample * scale

        return x_sample, y_sample