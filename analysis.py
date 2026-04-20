import pandas as pd
import json
import os
import glob
from src.evaluate import evaluate_results

def load_data():
    data = {}
    for f in glob.glob("manifests/*_data.json"):
        task = os.path.basename(f).replace("_data.json", "")
        with open(f, "r") as info:
            data[task] = json.load(info)
    return data

def main():
    data = load_data()
    
    # Load CSVs
    try:
        baseline = pd.read_csv("results/baseline_stochastic.csv")
        spec_stoch = pd.read_csv("results/spec_0.5B_k4_stoch.csv")
        spec_det = pd.read_csv("results/spec_0.5B_k4_det.csv")
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    # Evaluate Quality
    print("Baseline Stochastic Quality:")
    q_base = evaluate_results(baseline.to_dict('records'), data)
    print("\nSpec 0.5B k4 Stochastic Quality:")
    q_stoch = evaluate_results(spec_stoch.to_dict('records'), data)
    print("\nSpec 0.5B k4 Deterministic Quality:")
    q_det = evaluate_results(spec_det.to_dict('records'), data)

    # Performance Analysis - Group by Task
    def get_task_perf(df):
        # We need task column. Assuming it exists in the CSV based on evaluate_results using 'task'
        # Group by task
        perf = df.groupby('task').agg({
            'latency': 'sum',
            'alpha': 'mean',
            'b_eff': 'mean'
        }).rename(columns={'latency': 'total_latency', 'alpha': 'mean_alpha', 'b_eff': 'mean_b_eff'})
        return perf

    perf_base = baseline.groupby('task')['latency'].sum().rename('base_latency')
    perf_stoch = spec_stoch.groupby('task').agg({
        'latency': 'sum',
        'alpha': 'mean',
        'b_eff': 'mean'
    })
    
    perf_det = spec_det.groupby('task').agg({
        'latency': 'mean', # we will use b_eff and alpha primarily
        'alpha': 'mean',
        'b_eff': 'mean'
    })

    # Speedup: sum(base_latency) / sum(spec_latency) per task
    speedup = (perf_base / perf_stoch['latency']).rename('speedup')

    # Tables
    tasks = sorted(q_base.keys())
    
    # table 1: Quality (%)
    quality_df = pd.DataFrame({
        'Task': tasks,
        'Base': [q_base.get(t, 0) for t in tasks],
        'Spec Stoch': [q_stoch.get(t, 0) for t in tasks],
        'Spec Det': [q_det.get(t, 0) for t in tasks]
    })
    quality_df['Delta Stoch'] = quality_df['Spec Stoch'] - quality_df['Base']
    quality_df['Delta Det'] = quality_df['Spec Det'] - quality_df['Base']
    
    # table 2: Efficiency (Stochastic vs Baseline)
    eff_stoch = pd.DataFrame({
        'Task': tasks,
        'Speedup': [speedup.get(t, 0) for t in tasks],
        'Mean Alpha': [perf_stoch['alpha'].get(t, 0) for t in tasks],
        'Mean B_eff': [perf_stoch['b_eff'].get(t, 0) for t in tasks]
    })

    # table 3: Efficiency (Deterministic)
    eff_det = pd.DataFrame({
        'Task': tasks,
        'Mean Alpha': [perf_det['alpha'].get(t, 0) for t in tasks],
        'Mean B_eff': [perf_det['b_eff'].get(t, 0) for t in tasks]
    })

    print("\n--- QUALITY TABLE (%) ---")
    print(quality_df.to_string(index=False))
    
    print("\n--- PERFORMANCE (STOCHASTIC) ---")
    print(eff_stoch.to_string(index=False))
    
    print("\n--- PERFORMANCE (DETERMINISTIC) ---")
    print(eff_det.to_string(index=False))

if __name__ == "__main__":
    main()
