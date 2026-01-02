import os
import sys
import numpy as np
import mne
from pathlib import Path
import glob
from mne.decoding import CSP
from sklearn.model_selection import train_test_split

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
structure_path = os.path.join(os.path.dirname(current_dir), 'structure')
if structure_path not in sys.path: sys.path.append(structure_path)

from test_csp_baseline import UnifiedEEGPreprocessor

class FBCSP_Processor:
    def __init__(self, root_output_dir, n_csp_components=8, sfreq=128):
        self.root_output_dir = Path(root_output_dir)
        self.sfreq = sfreq
        self.n_components = n_csp_components 
        
        # === 构建 55 个频带 ===
        self.freq_bands = []
        for start in range(8, 29): self.freq_bands.append((start, start + 2))
        for start in range(8, 27): self.freq_bands.append((start, start + 4))
        for start in range(8, 23): self.freq_bands.append((start, start + 8))
        print(f"FBCSP 初始化: 55 个频带准备就绪")
        
        self.preprocessor = UnifiedEEGPreprocessor(target_rate=sfreq)
        self.csps = [CSP(n_components=n_csp_components, reg=None, log=True, norm_trace=False) 
                     for _ in range(len(self.freq_bands))]

    def _apply_filter_bank(self, X):
        n_trials, n_ch, n_time = X.shape
        X_filtered = np.zeros((n_trials, len(self.freq_bands), n_ch, n_time))
        for i, (l, h) in enumerate(self.freq_bands):
            try:
                X_filtered[:, i, :, :] = mne.filter.filter_data(
                    X.astype(np.float64), self.sfreq, l, h, method='iir', verbose=False
                )
            except: pass
        return X_filtered

    def process_and_save(self, file_path, dataset_type):
        file_path = str(file_path)
        filename = os.path.basename(file_path)
        
        # 1. 加载数据
        X, y = None, None
        try:
            if dataset_type == 'BCIC IV 2a':
                X, y = self.preprocessor.process_bcic_iv_2a(file_path)
            elif dataset_type == 'OpenBMI':
                X, y = self.preprocessor.process_openbmi(file_path)
            elif dataset_type == 'PhysioNet':
                # PhysioNet 需要特殊处理: S001R04.edf -> Subject S001
                # process_physionet 会自动找 R04,08,12
                X, y = self.preprocessor.process_physionet(file_path, runs=[4, 8, 12])
        except Exception as e:
            print(f"❌ 加载错误 {filename}: {e}")
            return
        
        if X is None or len(X) < 10:
            return # 静默跳过无效数据

        print(f"[处理中] {dataset_type} - {filename} (Samples: {len(X)})")

        # 2. 拆分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 3. FBCSP
        X_train_fb = self._apply_filter_bank(X_train)
        X_test_fb = self._apply_filter_bank(X_test)
        
        n_time = X_train.shape[-1]
        X_train_final = np.zeros((len(X_train), 55, self.n_components, n_time))
        X_test_final = np.zeros((len(X_test), 55, self.n_components, n_time))
        
        for i in range(55):
            x_tr_band = X_train_fb[:, i, :, :]
            x_te_band = X_test_fb[:, i, :, :]
            try:
                self.csps[i].fit(x_tr_band, y_train)
                filters = self.csps[i].filters_[:self.n_components] 
                X_train_final[:, i, :, :] = np.einsum('kc,bct->bkt', filters, x_tr_band)
                X_test_final[:, i, :, :] = np.einsum('kc,bct->bkt', filters, x_te_band)
            except: pass

        # 4. 转置 -> (Batch, Bands, Time, CSP)
        X_train_ready = X_train_final.transpose(0, 1, 3, 2)
        X_test_ready = X_test_final.transpose(0, 1, 3, 2)
        
        # 5. 保存
        # PhysioNet 命名特殊处理
        if dataset_type == 'PhysioNet':
            rel_path = filename.split('R')[0] # S001R04 -> S001
        else:
            rel_path = Path(file_path).stem

        save_dir = self.root_output_dir / dataset_type.replace(' ', '_') / rel_path
        os.makedirs(save_dir, exist_ok=True)

        np.save(save_dir / 'train_data.npy', {'X': X_train_ready, 'y': y_train})
        np.save(save_dir / 'test_data.npy',  {'X': X_test_ready,  'y': y_test})
        print(f" -> ✅ 保存: {save_dir}")

if __name__ == "__main__":
    OUTPUT_ROOT = r"D:\fyp\dataset_processed_fbcsp_all" 
    processor = FBCSP_Processor(root_output_dir=OUTPUT_ROOT)

    # 1. BCIC
    print("\n=== 扫描 BCIC IV 2a ===")
    bcic_dir = r"D:\fyp\dataset\BCIC_IV_2a\BCICIV_2a_gdf"
    for f in glob.glob(os.path.join(bcic_dir, "*.gdf")): 
        if "EOG" not in f: processor.process_and_save(f, 'BCIC IV 2a')

    # 2. OpenBMI
    print("\n=== 扫描 OpenBMI ===")
    openbmi_root = r"D:\fyp\dataset\openBMI\MNE-lee2019-mi-data\gigadb-datasets\live\pub\10.5524\100001_101000\100542"
    for root, dirs, files in os.walk(openbmi_root):
        for file in files:
            if file.endswith(".mat") and "EEG_MI" in file:
                processor.process_and_save(os.path.join(root, file), 'OpenBMI')

    # 3. PhysioNet
    print("\n=== 扫描 PhysioNet ===")
    physionet_root = r"D:\fyp\dataset\PhysioNet\MNE-eegbci-data\files\eegmmidb\1.0.0"
    for subj in sorted(os.listdir(physionet_root)):
        if subj.startswith('S'):
            target_file = os.path.join(physionet_root, subj, f"{subj}R04.edf")
            if os.path.exists(target_file):
                processor.process_and_save(target_file, 'PhysioNet')