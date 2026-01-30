import os
import sys
import numpy as np
import mne
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split

# Path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
structure_path = os.path.join(os.path.dirname(current_dir), 'structure')
if structure_path not in sys.path: sys.path.append(structure_path)

from test_csp_baseline import UnifiedEEGPreprocessor

class FilterBank_NoCSP_Processor:
    def __init__(self, root_output_dir, sfreq=128):
        self.root_output_dir = Path(root_output_dir)
        self.sfreq = sfreq
        
        # 1. Construct 55 frequency bands (Same as before)
        self.freq_bands = []
        for start in range(8, 29): self.freq_bands.append((start, start + 2))
        for start in range(8, 27): self.freq_bands.append((start, start + 4))
        for start in range(8, 23): self.freq_bands.append((start, start + 8))
        print(f"FilterBank initialized: 55 frequency bands ready (No CSP).")
        
        # 2. Unified Preprocessor (Ensure consistent 22 channels)
        self.preprocessor = UnifiedEEGPreprocessor(target_rate=sfreq)

    def _apply_filter_bank(self, X):
        """
        Input: (n_trials, n_ch, n_time)
        Output: (n_trials, n_bands, n_ch, n_time)
        """
        n_trials, n_ch, n_time = X.shape
        X_filtered = np.zeros((n_trials, len(self.freq_bands), n_ch, n_time))
        
        for i, (l, h) in enumerate(self.freq_bands):
            try:
                # Use MNE's IIR filter
                X_filtered[:, i, :, :] = mne.filter.filter_data(
                    X.astype(np.float64), self.sfreq, l, h, method='iir', verbose=False
                )
            except Exception as e:
                print(f"Filter error at band {l}-{h}Hz: {e}")
        return X_filtered

    def process_and_save(self, file_path, dataset_type):
        file_path = str(file_path)
        filename = os.path.basename(file_path)
        
        # 1. Load data using Unified Preprocessor
        # This guarantees we get the standard 22 channels (C3, C4, etc.)
        X, y = None, None
        try:
            if dataset_type == 'BCIC IV 2a':
                X, y = self.preprocessor.process_bcic_iv_2a(file_path)
            elif dataset_type == 'OpenBMI':
                X, y = self.preprocessor.process_openbmi(file_path)
            elif dataset_type == 'PhysioNet':
                X, y = self.preprocessor.process_physionet(file_path, runs=[4, 8, 12])
        except Exception as e:
            print(f"Loading error {filename}: {e}")
            return
        
        if X is None or len(X) < 10:
            return 

        print(f"[Processing] {dataset_type} - {filename} (Samples: {len(X)})")

        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 3. Filter Bank Only (No CSP Projection)
        # X shape: (Batch, 22, Time) -> X_fb shape: (Batch, 55, 22, Time)
        X_train_fb = self._apply_filter_bank(X_train)
        X_test_fb = self._apply_filter_bank(X_test)
        
        # 4. Transpose to align with model input
        # Target format: (Batch, Bands, Time, Channels)
        # Original: (Batch, Bands, Channels, Time) -> (Batch, Bands, Time, Channels)
        X_train_ready = X_train_fb.transpose(0, 1, 3, 2).astype(np.float32)
        X_test_ready = X_test_fb.transpose(0, 1, 3, 2).astype(np.float32)
        
        # 5. Save
        if dataset_type == 'PhysioNet':
            rel_path = filename.split('R')[0]
        else:
            rel_path = Path(file_path).stem

        save_dir = self.root_output_dir / dataset_type.replace(' ', '_') / rel_path
        os.makedirs(save_dir, exist_ok=True)

        np.save(save_dir / 'train_data.npy', {'X': X_train_ready, 'y': y_train})
        np.save(save_dir / 'test_data.npy',  {'X': X_test_ready,  'y': y_test})
        print(f" -> Saved: {save_dir} | Shape: {X_train_ready.shape}")

if __name__ == "__main__":
    # Define new output directory for No-CSP data
    OUTPUT_ROOT = r"D:\fyp\dataset_processed_no_csp" 
    processor = FilterBank_NoCSP_Processor(root_output_dir=OUTPUT_ROOT)

    # 1. BCIC IV 2a
    print("\nScanning BCIC IV 2a")
    bcic_dir = r"D:\fyp\dataset\BCIC_IV_2a\BCICIV_2a_gdf"
    for f in glob.glob(os.path.join(bcic_dir, "*.gdf")): 
        if "EOG" not in f: processor.process_and_save(f, 'BCIC IV 2a')

    # 2. OpenBMI
    print("\nScanning OpenBMI")
    openbmi_root = r"D:\fyp\dataset\openBMI\MNE-lee2019-mi-data\gigadb-datasets\live\pub\10.5524\100001_101000\100542"
    for root, dirs, files in os.walk(openbmi_root):
        for file in files:
            if file.endswith(".mat") and "EEG_MI" in file:
                processor.process_and_save(os.path.join(root, file), 'OpenBMI')

    # 3. PhysioNet
    print("\nScanning PhysioNet")
    physionet_root = r"D:\fyp\dataset\PhysioNet\MNE-eegbci-data\files\eegmmidb\1.0.0"
    for subj in sorted(os.listdir(physionet_root)):
        if subj.startswith('S'):
            target_file = os.path.join(physionet_root, subj, f"{subj}R04.edf")
            if os.path.exists(target_file):
                processor.process_and_save(target_file, 'PhysioNet')