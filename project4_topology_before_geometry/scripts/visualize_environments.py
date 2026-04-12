"""Script to visualize all registered environments in project4."""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from project4_topology_before_geometry.environments.env_factory import list_environments, get_env

def main():
    output_dir = os.path.join(project_root, "project4_topology_before_geometry", "figures", "env_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    envs = list_environments()
    print(f"Found {len(envs)} environments: {envs}")
    
    for env_name in envs:
        print(f"Visualizing {env_name}...")
        try:
            env = get_env(env_name)
            
            plt.figure(figsize=(8, 8))
            
            if env.backend == "minigrid":
                # For minigrid, we can get a high-quality render of the grid
                inner_env = env.make_env()
                # We need to reset the environment to initialize the grid and agent
                inner_env.reset()
                # get_frame() returns an RGB image of the whole grid
                img = inner_env.get_frame(tile_size=32)
                plt.imshow(img)
                plt.title(f"{env_name} (MiniGrid)")
            else:
                # For RatInABox or others, we plot the traversable mask
                mask = env.traversable_mask
                plt.imshow(mask, cmap='gray', origin='lower')
                plt.title(f"{env_name} ({env.backend})")
            
            plt.axis('off')
            save_path = os.path.join(output_dir, f"{env_name}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"  Saved to {save_path}")
            
        except Exception as e:
            import traceback
            print(f"  Failed to visualize {env_name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
