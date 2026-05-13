
import os
import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(os.getcwd())

from project4_topology_before_geometry.scripts.run_local import main

# Patch the config path in the main function's scope if possible, 
# or just copy run_local but with a different config path.

import project4_topology_before_geometry.scripts.run_local as run_local

def patched_main():
    # Force use of test_config.yaml
    original_cfg_path = Path(run_local.__file__).resolve().parent.parent / "configs" / "local_config.yaml"
    test_cfg_path = Path(run_local.__file__).resolve().parent.parent / "configs" / "test_config.yaml"
    
    # We can't easily patch the local variable inside main, so let's just re-implement main briefly here
    import torch
    import numpy as np
    from project4_topology_before_geometry.environments.env_factory import get_env
    from project4_topology_before_geometry.environments.topology_labels import TOPOLOGY_LABELS
    from project4_topology_before_geometry.models.objectives import LossFactory
    from project4_topology_before_geometry.models.prnn import RolloutPRNN
    from project4_topology_before_geometry.sensory.action_encoder import ActionEncoder
    from project4_topology_before_geometry.training.trainer import Trainer
    from project4_topology_before_geometry.evaluation.convergence_tracker import ConvergenceTracker

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu") # Use CPU for test
    
    cfg = run_local._load_config(test_cfg_path)
    cfg["device"] = str(device)
    for directory in (cfg["checkpoint_dir"], cfg["log_dir"], cfg["figures_dir"]):
        Path(directory).mkdir(parents=True, exist_ok=True)

    for env_name in cfg["environments"]:
        print(f"Running {env_name} test...")
        env = get_env(env_name, cfg)
        act_enc = ActionEncoder(backend=cfg["env_backend"][env_name])
        model = RolloutPRNN(
            obs_dim=env.obs_dim,
            act_dim=act_enc.act_dim,
            hidden_dim=cfg["hidden_dim"],
            rollout_k=cfg["rollout_k"],
            device=device,
        )
        loss_fn = LossFactory.get_loss(cfg["loss_type"], cfg["rollout_k"], device)
        tracker = ConvergenceTracker(env, model, TOPOLOGY_LABELS[env_name], cfg)
        trainer = Trainer(cfg, env, model, act_enc, loss_fn, tracker, device=device)
        trainer.train(n_trials=cfg["n_trials"])
        results = tracker.finalize()
        print(f"Test finished: {results}")

if __name__ == "__main__":
    patched_main()
