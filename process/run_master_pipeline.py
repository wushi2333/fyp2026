import subprocess
import sys
import os
import time
from datetime import timedelta

def run_script(script_name, step_description):
    """
    Run a Python script in a separate subprocess.
    """
    # Get the absolute path of the currently running script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)
    
    print(f"\n{'='*80}")
    print(f" [Pipeline running] {step_description}")
    print(f" Running script: {script_name}")
    print(f"{'='*80}\n")
    
    if not os.path.exists(script_path):
        print(f" [Error] File not found: {script_path}")
        print("Please check the filename or ensure it is located in the process directory!")
        sys.exit(1)
        
    start_time = time.time()
    
    try:
        # Launch the subprocess and stream the output to the console
        # sys.executable refers to the current Python interpreter (ensures the virtual environment is correct)
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\n [Fatal Error] Script {script_name} failed. Pipeline aborted.")
        sys.exit(e.returncode)
        
    elapsed = time.time() - start_time
    print(f"\n[Completed] {script_name} finished! Elapsed time: {timedelta(seconds=int(elapsed))}")

def main():
    pipeline_start = time.time()
    
    print(f" {'*'*74} ")
    print(f" {' '*18} MI-EEG automated experiment pipeline started {' '*18} ")
    print(f" {'*'*74} ")
    
    # Define the list of scripts to run in order
    # Format: (script filename, step description)
    pipeline_scripts = [
        (
            "train_baseline_dl_20runs.py",  # <--- Ensure this baseline script has this filename, or modify as needed
            "Step 1: Run baseline models (EEGNet & DeepConvNet 20 Runs)"
        ),
        (
            "train_experiment_20runs.py", 
            "Step 2: Run proposed model (MoE FBCSP 20 Runs)"
        ),
        (
            "analyze_statistics_multi.py", 
            "Step 3: Aggregate results and generate statistical significance summary table (T-test)"
        )
    ]
    
    # Execute sequentially
    for script_name, description in pipeline_scripts:
        run_script(script_name, description)
        
    # Summary output
    total_time = time.time() - pipeline_start
    print(f"\n {'='*76} ")
    print(f" All experiment tasks completed successfully!")
    print(f" Total pipeline elapsed time: {timedelta(seconds=int(total_time))}")
    print(f" The final academic summary table has been saved to:")
    print(f"   C:\\Users\\巫逝\\Desktop\\学习\\大四\\毕设\\code\\final_year_project\\result\\statistical_analysis")
    print(f" {'='*76} ")

if __name__ == "__main__":
    main()