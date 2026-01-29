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
        
        # S&R parameters
        self.snr_aug = snr_aug and (mode == 'train')
        self.snr_prob = snr_prob
        self.num_segments = num_segments
        
        #Keep a list of fragments to avoid memory spikes
        self.X_chunks = [] 
        self.y_chunks = []
        self.chunk_sizes = []
        self.cumulative_sizes = []
        
        target_file = 'train_data.npy' if mode == 'train' else 'test_data.npy'
        print(f"[{mode.upper()}] Scanning data (Target: {target_dataset})...")
        print(f" -> Tip: Memory optimization mode enabled (List-based storage)")
        
        total_samples = 0
        loaded_files = 0
        
        for root, dirs, files in os.walk(root_dir):
            if target_dataset != 'All' and target_dataset not in root:
                continue
                
            if target_file in files:
                file_path = os.path.join(root, target_file)
                try:
                    # Load data
                    data = np.load(file_path, allow_pickle=True).item()
                    X_part = data['X']
                    y_part = data['y']
                    
                    # Immediately convert to float32 to save half the memory
                    if X_part.dtype != np.float32:
                        X_part = X_part.astype(np.float32)
                        
                    # Store fragments, do not merge
                    self.X_chunks.append(X_part)
                    self.y_chunks.append(torch.from_numpy(y_part).long())
                    
                    count = len(X_part)
                    self.chunk_sizes.append(count)
                    total_samples += count
                    loaded_files += 1
                    
                    # Print progress to prevent freezing
                    if loaded_files % 10 == 0:
                        print(f"    ...Loaded {loaded_files} files ({total_samples} samples)")
                        
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

        # Build a cumulative index for quick positioning
        self.cumulative_sizes = np.cumsum(self.chunk_sizes)
        self.total_len = total_samples
        
        if total_samples > 0:
            print(f" -> Loading complete: {loaded_files} files, {total_samples} samples in total")
            
            # S&R index construction
            if self.snr_aug:
                print(" -> Building S&R global index (this may take a few seconds)...")
                self.global_labels = torch.cat(self.y_chunks) 
                self.class_indices = {}
                unique_labels = torch.unique(self.global_labels).numpy()
                for label in unique_labels:
                    self.class_indices[label] = (self.global_labels == label).nonzero(as_tuple=True)[0]
                print(" -> S&R index construction complete")
        else:
            print(f"Warning: No data found!")

    def __len__(self):
        return self.total_len

    def _get_chunk_index(self, global_idx):
        """
        Find (Chunk index, offset within Chunk) based on global index
        Use binary search to speed up
        """
        chunk_idx = bisect.bisect_right(self.cumulative_sizes, global_idx)
        if chunk_idx == 0:
            local_idx = global_idx
        else:
            local_idx = global_idx - self.cumulative_sizes[chunk_idx - 1]
        return chunk_idx, local_idx

    def _get_data_at(self, global_idx):
        chunk_idx, local_idx = self._get_chunk_index(global_idx)
        # Convert Numpy to Tensor only at this time to reduce resident memory overhead
        x = torch.from_numpy(self.X_chunks[chunk_idx][local_idx])
        y = self.y_chunks[chunk_idx][local_idx]
        return x, y

    def _apply_snr(self, x_current, y_current):

        label = y_current.item()
        candidates = self.class_indices[label]
        
        n_bands, n_time, n_csp = x_current.shape
        segment_len = n_time // self.num_segments
        x_new = torch.zeros_like(x_current)
        
        for i in range(self.num_segments):
            start = i * segment_len
            end = start + segment_len if i < self.num_segments - 1 else n_time
            
            # Randomly select a global index
            random_global_idx = candidates[torch.randint(0, len(candidates), (1,)).item()].item()
            
            # Get the source sample
            chunk_idx, local_idx = self._get_chunk_index(random_global_idx)
            x_source_np = self.X_chunks[chunk_idx][local_idx]
            # Temporarily convert to Tensor
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