import os
import pandas as pd
import numpy as np
from scipy import stats

# Force save path
SAVE_DIR = r"C:\Users\巫逝\Desktop\学习\大四\毕设\code\final_year_project\result\statistical_analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

def perform_multi_baseline_analysis(proposed_csv, baselines_dict):
    print(f"{'='*80}")
    print("Multi-baseline significance analysis (Paired t-test)")
    print(f"{'='*80}")
    
    if not os.path.exists(proposed_csv):
        raise FileNotFoundError(f"Proposed model data not found: {proposed_csv}")
        
    df_prop = pd.read_csv(proposed_csv)
    subjects = df_prop['Subject'].unique()
    
    # Verify baseline files exist and load them
    loaded_baselines = {}
    for bl_name, bl_path in baselines_dict.items():
        if os.path.exists(bl_path):
            loaded_baselines[bl_name] = pd.read_csv(bl_path)
            print(f" -> Successfully loaded baseline: {bl_name}")
        else:
            print(f" -> [Warning] Baseline data not found: {bl_name} (path: {bl_path})")

    report_data = []
    
    # Lists used to compute global averages
    global_prop_accs = []
    global_base_accs = {bl_name: [] for bl_name in loaded_baselines.keys()}
    
    for subj in subjects:
        # 1. Retrieve Proposed model data
        prop_accs = df_prop[df_prop['Subject'] == subj]['Acc'].values
        global_prop_accs.extend(prop_accs)
        prop_mean, prop_std = np.mean(prop_accs), np.std(prop_accs)
        
        row_dict = {
            'Subject': subj,
            'Proposed_Acc(%)': f"{prop_mean:.2f} ± {prop_std:.2f}"
        }
        
        # 2. Retrieve each baseline's data and compare
        for bl_name, df_base in loaded_baselines.items():
            base_accs = df_base[df_base['Subject'] == subj]['Acc'].values
            
            # Ensure sample count alignment (usually 20 runs)
            if len(base_accs) == len(prop_accs):
                global_base_accs[bl_name].extend(base_accs)
                base_mean, base_std = np.mean(base_accs), np.std(base_accs)
                
                # Compute paired t-test p-value
                t_stat, p_val = stats.ttest_rel(prop_accs, base_accs)
                
                # Significance annotation
                sig_mark = ""
                if p_val < 0.01: sig_mark = "**"
                elif p_val < 0.05: sig_mark = "*"
                
                row_dict[f'{bl_name}_Acc(%)'] = f"{base_mean:.2f} ± {base_std:.2f}"
                row_dict[f'P-val ({bl_name})'] = f"{p_val:.4f} {sig_mark}"
            else:
                row_dict[f'{bl_name}_Acc(%)'] = "N/A"
                row_dict[f'P-val ({bl_name})'] = "N/A"
                
        report_data.append(row_dict)

    # 3. Compute the overall average statistics
    avg_row = {'Subject': 'AVERAGE'}
    avg_row['Proposed_Acc(%)'] = f"{np.mean(global_prop_accs):.2f} ± {np.std(global_prop_accs):.2f}"
    
    for bl_name in loaded_baselines.keys():
        b_accs = global_base_accs[bl_name]
        if len(b_accs) == len(global_prop_accs) and len(b_accs) > 0:
            avg_row[f'{bl_name}_Acc(%)'] = f"{np.mean(b_accs):.2f} ± {np.std(b_accs):.2f}"
            t_stat, p_val = stats.ttest_rel(global_prop_accs, b_accs)
            sig_mark = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
            avg_row[f'P-val ({bl_name})'] = f"{p_val:.4f} {sig_mark}"
            
    report_data.append(avg_row)

    # 4. Generate and save the report
    df_report = pd.DataFrame(report_data)
    
    # Reorder columns for a paper-style layout
    cols = ['Subject', 'Proposed_Acc(%)']
    for bl_name in loaded_baselines.keys():
        cols.append(f'{bl_name}_Acc(%)')
        cols.append(f'P-val ({bl_name})')
    df_report = df_report[cols]
    
    save_path = os.path.join(SAVE_DIR, "thesis_ready_multi_baselines.csv")
    df_report.to_csv(save_path, index=False)
    
    print("\n" + df_report.to_string(index=False))
    print(f"\n>>> Academic summary table saved to: {save_path} <<<")
    print("Note: * indicates p < 0.05, ** indicates p < 0.01")


if __name__ == "__main__":
    # 1. Provide the absolute path to the proposed model's 20-run results CSV
    PROPOSED_CSV = r"C:\Users\巫逝\Desktop\学习\大四\毕设\code\final_year_project\result\new_experiment_20runs\no_transfer\raw_20runs_metrics.csv"
    
    # 2. Provide the baseline models CSV path dictionary
    BASELINES_DICT = {
        "EEGNet": r"C:\Users\巫逝\Desktop\学习\大四\毕设\code\final_year_project\result\baselines\EEGNet\EEGNet_raw_20runs.csv",
        "DeepConvNet": r"C:\Users\巫逝\Desktop\学习\大四\毕设\code\final_year_project\result\baselines\DeepConvNet\DeepConvNet_raw_20runs.csv",

    }
    
    perform_multi_baseline_analysis(PROPOSED_CSV, BASELINES_DICT)