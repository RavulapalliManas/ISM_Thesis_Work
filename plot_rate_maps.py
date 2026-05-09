import numpy as np
import matplotlib.pyplot as plt
import os

def plot_rate_maps(out_dir="./hd_ablation_results", step=300):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    modes = ['full', 'ablated']
    for row, mode in enumerate(modes):
        path = os.path.join(out_dir, f"H_{mode}_s0_step{step}.npy")
        if not os.path.exists(path):
            print(f"Skipping {mode}, no H matrix found at {path}")
            continue
            
        H = np.load(path) # (324, 128)
        
        # Pick top 4 units by variance
        variances = np.var(H, axis=0)
        top_units = np.argsort(variances)[-4:]
        
        for col, unit_idx in enumerate(top_units):
            ax = axes[row, col]
            if H.shape[0] == 324:
                grid = H[:, unit_idx].reshape(18, 18)
            else:
                s = int(np.sqrt(H.shape[0]))
                grid = H[:s*s, unit_idx].reshape(s, s)
                
            ax.imshow(grid, cmap='viridis', origin='lower')
            ax.set_title(f"{mode.capitalize()} Unit {unit_idx}")
            ax.axis('off')
            
    plt.tight_layout()
    save_path = "project5_symmetry/Report/rate_maps.pdf"
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    plot_rate_maps()
