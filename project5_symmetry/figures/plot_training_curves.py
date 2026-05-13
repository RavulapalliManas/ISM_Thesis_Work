import json
import matplotlib.pyplot as plt
import os

def plot_curves(out_dir="./hd_ablation_results", step=300):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'full': '#2ecc71', 'ablated': '#e74c3c'}
    
    for mode in ['full', 'ablated']:
        path = os.path.join(out_dir, f"log_{mode}_s0.json")
        if not os.path.exists(path):
            print(f"Skipping {mode}, no log found.")
            continue
            
        with open(path, 'r') as f:
            log = json.load(f)
            
        ax1.plot(log['steps'], log['losses'], label=mode.capitalize(), color=colors[mode], lw=2)
        ax2.plot(log['steps'], log['srsa'], label=mode.capitalize(), color=colors[mode], lw=2)
        
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("sRSA")
    ax2.set_title("Representational Similarity")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    
    plt.tight_layout()
    save_path = "project5_symmetry/Report/training_curves.pdf"
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    plot_curves()
