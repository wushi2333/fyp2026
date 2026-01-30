import os
import numpy as np
import mne
import scipy.io
import warnings
from typing import Tuple, Optional, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import re
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")

class UnifiedEEGPreprocessor:
    def __init__(self, target_rate: int = 128, trial_duration: float = 4.0):
        self.target_rate = target_rate
        self.trial_duration = trial_duration
        
        self.target_channels = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 
            'P1', 'Pz', 'P2', 'POz'
        ]

    def _apply_advanced_filters(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw = raw.copy()
        raw.notch_filter(freqs=[50, 60], verbose=False)
        raw.filter(l_freq=0.5, h_freq=45.0, fir_design='firwin', verbose=False)
        return raw

    def _robust_z_score(self, data: np.ndarray) -> np.ndarray:
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        return (data - mean) / (std + 1e-8)

    def augment_dataset(self, X: np.ndarray, y: np.ndarray, method='noise', factor=1) -> Tuple[np.ndarray, np.ndarray]:
        X_aug_list = [X]
        y_aug_list = [y]
        print(f"  -> Performing data augmentation (Strategy: {method}, Factor: {factor}x)...")
        for _ in range(factor):
            if method == 'noise':
                noise_level = 0.05
                noise = np.random.normal(0, noise_level, X.shape)
                X_new = X + noise
            elif method == 'scaling':
                scales = np.random.uniform(0.8, 1.2, size=(X.shape[0], 1, 1))
                X_new = X * scales
            else:
                continue
            X_aug_list.append(X_new)
            y_aug_list.append(y)
        return np.concatenate(X_aug_list, axis=0), np.concatenate(y_aug_list, axis=0)

    def _normalize_channel_names(self, raw: mne.io.Raw) -> Dict[str, str]:
        mapping = {}
        for ch in raw.ch_names:
            clean_name = ch.strip().strip('.').upper()
            if clean_name == 'FZ': target = 'Fz'
            elif clean_name == 'FCZ': target = 'FCz'
            elif clean_name == 'CZ': target = 'Cz'
            elif clean_name == 'CPZ': target = 'CPz'
            elif clean_name == 'PZ': target = 'Pz'
            elif clean_name == 'POZ': target = 'POz'
            else:
                target = next((t for t in self.target_channels if t.upper() == clean_name), None)
            if target:
                mapping[ch] = target
        return mapping

    def process_bcic_iv_2a(self, file_path: str, augment: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        print(f"Processing BCIC IV 2a: {os.path.basename(file_path)}")
        try:
            raw = mne.io.read_raw_gdf(file_path, preload=True)
            current_names = raw.ch_names
            if len(current_names) < 22: return None, None
            rename_map = {current_names[i]: self.target_channels[i] for i in range(22)}
            raw.rename_channels(rename_map)
            raw.pick_channels(self.target_channels)
            raw = self._apply_advanced_filters(raw)
            raw.set_eeg_reference('average', projection=False)
            if raw.info['sfreq'] != self.target_rate:
                raw.resample(self.target_rate)
            events, annot_map = mne.events_from_annotations(raw)
            event_id = {'Left': annot_map.get('769'), 'Right': annot_map.get('770')}
            if None in event_id.values(): return None, None
            epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=self.trial_duration, baseline=None, preload=True, verbose=False)
            X = epochs.get_data()[:, :, :int(self.target_rate * self.trial_duration)]
            y = epochs.events[:, -1]
            y_mapped = np.array([0 if label == event_id['Left'] else 1 for label in y])
            X_norm = np.array([self._robust_z_score(x) for x in X])
            if augment:
                X_norm, y_mapped = self.augment_dataset(X_norm, y_mapped)
            return X_norm, y_mapped
        except Exception as e:
            print(f"Error: {e}")
            return None, None

    def process_openbmi(self, file_path: str, augment: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        print(f"Processing OpenBMI: {os.path.basename(file_path)}")
        try:
            # 1. Load .mat
            mat = scipy.io.loadmat(file_path)
            data_struct = mat['EEG_MI_train'] if 'EEG_MI_train' in mat else mat['EEG_MI_test']
            struct = data_struct[0, 0]
            X_continuous = struct['x']
            t_trials = struct['t'].flatten()
            y_labels = struct['y_dec'].flatten()
            
            # 2. Build MNE Raw
            openbmi_chans = [
                'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 
                'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 
                'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 
                'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 
                'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 
                'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'
            ]
            info = mne.create_info(ch_names=openbmi_chans, sfreq=1000, ch_types='eeg')
            raw = mne.io.RawArray(X_continuous.T, info)
            
            # 3. Restore reference electrode FCz and standardize channels
            raw.add_reference_channels(ref_channels=['FCz'])
            raw.rename_channels(self._normalize_channel_names(raw))
            raw.pick_channels(self.target_channels)
            
            # 4. Filter
            raw = self._apply_advanced_filters(raw)
            
            # 5. CAR
            raw.set_eeg_reference('average', projection=False)
            
            # 6. Resample
            if raw.info['sfreq'] != self.target_rate:
                raw.resample(self.target_rate)
                
            # 7. Epoching
            new_t_trials = (t_trials / 1000 * self.target_rate).astype(int)
            events = np.zeros((len(new_t_trials), 3), dtype=int)
            events[:, 0] = new_t_trials
            events[:, 2] = y_labels
            # OpenBMI: 1=Left, 2=Right
            epochs = mne.Epochs(raw, events, event_id={'Left': 1, 'Right': 2}, tmin=0, tmax=self.trial_duration, baseline=None, preload=True, verbose=False)
            
            X = epochs.get_data()[:, :, :int(self.target_rate * self.trial_duration)]
            y = epochs.events[:, -1]
            y_mapped = np.array([0 if label == 1 else 1 for label in y])
            
            # 8. Normalization
            X_norm = np.array([self._robust_z_score(x) for x in X])
            
            if augment:
                X_norm, y_mapped = self.augment_dataset(X_norm, y_mapped)
                
            return X_norm, y_mapped
        except Exception as e:
            print(f"Error in OpenBMI: {e}")
            return None, None

    def process_physionet(self, file_path: str, augment: bool = False, runs: List[int] = [4, 8, 12]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        [Modified version] Process PhysioNet dataset, supporting multi-run merging.
        By default, Runs 4, 8, 12 (Left vs Right Fist task) are merged to increase the number of samples.
        Runs 6, 10, 14 are Fists vs Feet, which need to be processed separately or modify the Label Mapping if you want to use them.
        """
        # Parse Subject ID from filename (e.g. S001)
        base_dir = os.path.dirname(file_path)
        basename = os.path.basename(file_path)
        subject_match = re.search(r"(S\d{3})", basename)
        
        if not subject_match:
            print(f"Error: Cannot parse Subject ID (e.g. S001) from filename {basename}")
            return None, None
            
        subject_id = subject_match.group(1)
        print(f"Processing PhysioNet Subject: {subject_id}, merging Runs: {runs}")
        
        raw_list = []
        
        try:
            # Iterate through all required Runs
            for run_id in runs:
                # Construct filename: S001R04.edf, S001R08.edf ...
                run_filename = f"{subject_id}R{run_id:02d}.edf"
                run_path = os.path.join(base_dir, run_filename)
                
                if not os.path.exists(run_path):
                    print(f"  -> Warning: File not found {run_filename}, skipping.")
                    continue
                
                print(f"  -> Loading Run {run_id}: {run_filename}")
                raw = mne.io.read_raw_edf(run_path, preload=True, verbose=False)
                
                # Preprocess each Run
                raw.rename_channels(self._normalize_channel_names(raw))
                raw.pick_channels(self.target_channels)
                raw = self._apply_advanced_filters(raw) # Filter
                
                raw_list.append(raw)
            
            if not raw_list:
                print("Error: No valid Run data loaded.")
                return None, None
                
            # Merge all Raw objects
            if len(raw_list) > 1:
                raw_combined = mne.concatenate_raws(raw_list)
                print(f"  -> Merged {len(raw_list)} Runs.")
            else:
                raw_combined = raw_list[0]
            
            # Continue with subsequent processing
            raw_combined.set_eeg_reference('average', projection=False)
            raw_combined.resample(self.target_rate)
            
            events, annot_map = mne.events_from_annotations(raw_combined)
            
            # PhysioNet T1/T2 definition:
            # Runs 4, 8, 12: T1=Left Fist, T2=Right Fist
            # Runs 6, 10, 14: T1=Both Fists, T2=Both Feet
            # Here we assume we are only processing Left/Right tasks
            t1, t2 = annot_map.get('T1'), annot_map.get('T2')
            
            if t1 is None or t2 is None: 
                print("Warning: T1 or T2 events not found in combined data.")
                return None, None
            
            # Epoching
            epochs = mne.Epochs(raw_combined, events, {'Left': t1, 'Right': t2}, 
                                tmin=0, tmax=self.trial_duration, 
                                baseline=None, preload=True, verbose=False)
            
            X = epochs.get_data()[:, :, :int(self.target_rate * self.trial_duration)]
            y = epochs.events[:, -1]
            y_mapped = np.array([0 if label == t1 else 1 for label in y])
            
            X_norm = np.array([self._robust_z_score(x) for x in X])
            
            if augment:
                X_norm, y_mapped = self.augment_dataset(X_norm, y_mapped)
                
            return X_norm, y_mapped
            
        except Exception as e:
            print(f"Error in PhysioNet processing: {e}")
            return None, None

class CSPProcessor:
    def __init__(self, n_components=8, rank='info', log=True, norm_trace=False):
        self.n_components = n_components
        self.csp = CSP(n_components=n_components, reg=None, log=log, 
                       rank=rank, norm_trace=norm_trace)
        self.classifier = LogisticRegression(solver='liblinear')
        self.is_fitted = False

    def train_baseline(self, X_train, y_train, X_test, y_test):
        print(f"\n[ML Baseline] Fitting CSP (n_components={self.n_components})...")
        X_train_features = self.csp.fit_transform(X_train, y_train)
        self.is_fitted = True
        X_test_features = self.csp.transform(X_test)
        
        print("[ML Baseline] Training Logistic Regression classifier...")
        self.classifier.fit(X_train_features, y_train)
        
        y_pred = self.classifier.predict(X_test_features)
        acc = accuracy_score(y_test, y_pred)
        
        print("-" * 50)
        print(f"Machine Learning Baseline Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print("-" * 50)
        print(classification_report(y_test, y_pred, target_names=['Left', 'Right']))
        return acc

    def project_signals(self, X):
        if not self.is_fitted:
            raise ValueError("CSP has not been fitted yet")
        filters = self.csp.filters_[:self.n_components]
        X_projected = np.einsum('kc,bct->bkt', filters, X)
        return X_projected


class FourDFeatureExtractor:
    def __init__(self, sfreq: int = 128, n_components: int = 8):
        self.sfreq = sfreq
        self.n_components = n_components
        self.freq_bands = {
            'Delta': (0.5, 4), 'Theta': (4, 8),
            'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 45)
        }
        self.grid_h = 1
        self.grid_w = n_components

    def apply_filter_bank(self, x_data: np.ndarray) -> np.ndarray:
        n_batch, n_ch, n_time = x_data.shape
        x_filtered = np.zeros((n_batch, len(self.freq_bands), n_ch, n_time))
        for i, (band, (l, h)) in enumerate(self.freq_bands.items()):
            x_filtered[:, i, :, :] = mne.filter.filter_data(x_data, self.sfreq, l, h, method='iir', verbose=False)
        return x_filtered

    def map_to_spatial_grid(self, x_multiband: np.ndarray) -> np.ndarray:
        """
        Reshape multi-band CSP features into a 4D structure (removing the redundant Height dimension)
        Input x_multiband: (Batch, n_bands, n_components, Time)
        Target output: (Batch, n_bands, Time, n_components)
        """
        # (Batch, n_bands, n_components, Time) -> (Batch, n_bands, Time, n_components)
        x_permuted = np.transpose(x_multiband, (0, 1, 3, 2))

        return x_permuted

    def visualize_preprocessing(self, x_csp, x_multiband, x_grid, trial_idx=0):
        time_axis = np.linspace(0, x_csp.shape[-1]/self.sfreq, x_csp.shape[-1])
        
        plt.figure(figsize=(12, 12)) 
        plt.suptitle(f"4D Feature Visualization (CSP Projected) - Trial {trial_idx}", fontsize=16)
        
        plt.subplot(3, 1, 1)
        plt.plot(time_axis, x_csp[trial_idx, 0, :], 'k', linewidth=1.5, label='CSP Comp 0 (Broadband)')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, (band_name, _) in enumerate(self.freq_bands.items()):
             plt.plot(time_axis, x_multiband[trial_idx, i, 0, :], color=colors[i], alpha=0.6, linewidth=1, label=band_name)

        plt.title(f"Time Domain: CSP Component 0 (All 5 Bands included)")
        plt.legend(loc='upper right', ncol=3) 
        plt.xlabel("Time (s)")
        
        time_point = int(2.0 * self.sfreq)
        grid_snapshot = x_grid[trial_idx, :, time_point, :]
        
        plt.subplot(3, 1, 2)
        sns.heatmap(grid_snapshot, annot=True, fmt=".2f", cmap="viridis", 
                   xticklabels=[f"CSP{i}" for i in range(self.n_components)],
                   yticklabels=list(self.freq_bands.keys()))
        plt.title(f"Multi-band Energy Distribution (t=2.0s)\nRows: Bands, Cols: CSP Components")
        plt.xlabel("Spatial Components (CSP)")

        plt.subplot(3, 1, 3)
        band_energies = np.mean(np.abs(x_multiband[trial_idx]), axis=(1, 2))
        plt.bar(list(self.freq_bands.keys()), band_energies, color=colors)
        plt.title("Average Energy per Frequency Band (CSP Component 0)")
        plt.ylabel("Mean Absolute Amplitude")
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 1. Initialization
    n_csp_components = 8
    processor = UnifiedEEGPreprocessor(target_rate=128)
    csp_processor = CSPProcessor(n_components=n_csp_components)
    feature_extractor = FourDFeatureExtractor(sfreq=128, n_components=n_csp_components)

    paths = {
        'BCIC IV 2a': r"D:\fyp\dataset\BCIC_IV_2a\BCICIV_2a_gdf\A03T.gdf",
        'OpenBMI':    r"D:\fyp\dataset\openBMI\MNE-lee2019-mi-data\gigadb-datasets\live\pub\10.5524\100001_101000\100542\session1\s4\sess01_subj04_EEG_MI.mat",
        # PhysioNet only needs to provide any file under this Subject, and it will automatically find R04, R08, and R12 in the same directory
        'PhysioNet':  r"D:\fyp\dataset\PhysioNet\MNE-eegbci-data\files\eegmmidb\1.0.0\S001\S001R08.edf" 
    }

    print("Starting Multi-dataset CSP Benchmark Test + 4D Feature Reconstruction")

    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: File not found ({path})")
            continue
            
        print(f"\nProcessing {name}...")
        
        # 1. Call different processing functions based on the dataset name
        X, y = None, None
        if 'BCIC' in name:
            X, y = processor.process_bcic_iv_2a(path, augment=False)
        elif 'OpenBMI' in name:
            X, y = processor.process_openbmi(path, augment=False)
        elif 'PhysioNet' in name:
            X, y = processor.process_physionet(path, augment=False, runs=[4, 8, 12])

        if X is not None:
            # 2. Check the number of samples
            n_trials = X.shape[0]
            print(f"Data loaded successfully: Shape={X.shape} (Trials, Channels, Time)")
            
            if n_trials < 20:
                print(f"Warning: The number of samples is too small ({n_trials}), skipping CSP training for this dataset.")
                continue

            # 3. Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 4. Run benchmark test 
            print(f"Training CSP model (Dataset: {name})...")
            csp_processor = CSPProcessor(n_components=n_csp_components) 
            acc = csp_processor.train_baseline(X_train, y_train, X_test, y_test)
            
            # 5. Generate CSP projected data
            X_train_csp = csp_processor.project_signals(X_train)
            print(f"[Step 1] CSP projection completed: {X_train_csp.shape}")
            
            # 6. 4D feature construction
            X_bands = feature_extractor.apply_filter_bank(X_train_csp)
            print(f"[Step 2] Multi-band filtering completed: {X_bands.shape}")
            
            X_final_grid = feature_extractor.map_to_spatial_grid(X_bands)
            print(f"[Step 3] 4D grid reshaping completed: {X_final_grid.shape}")
            
            # 7. Visualization (only show the first sample)
            feature_extractor.visualize_preprocessing(X_train_csp, X_bands, X_final_grid, trial_idx=0)
            
        else:
            print(f"Failed to load {name} or task type mismatch.")