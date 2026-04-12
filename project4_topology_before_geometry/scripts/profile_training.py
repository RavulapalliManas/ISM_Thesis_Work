
import time
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from project4_topology_before_geometry.environments.env_factory import get_env
from project4_topology_before_geometry.models.prnn import RolloutPRNN
from project4_topology_before_geometry.sensory.action_encoder import ActionEncoder
from project4_topology_before_geometry.models.objectives import LossFactory
from project4_topology_before_geometry.evaluation.convergence_tracker import ConvergenceTracker

def profile_training():
    cfg = {
        "hidden_dim": 256,
        "rollout_k": 5,
        "seq_duration": 100,
        "n_trials": 200,  # Small number for profiling
        "eval_every_trials": 50,
        "topo_eval_every_trials": 100,
        "loss_type": "rollout_mse",
        "recurrence_scale": 1.0,
        "sigma_noise": 0.03,
        "dropout": 0.15,
        "neural_timescale": 2,
        "time_mode": "continuous",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "topo_device": "cpu",
        "mixed_precision": False,
        "n_accumulate": 1,
        "gradient_clip": 1.0,
        "environments": ["square_low_alias"],
        "env_backend": {"square_low_alias": "minigrid"},
        "log_dir": "project4_topology_before_geometry/logs_profile/",
        "checkpoint_dir": "project4_topology_before_geometry/checkpoints_profile/",
        "figures_dir": "project4_topology_before_geometry/figures_profile/"
    }
    
    device = torch.device(cfg["device"])
    print(f"Profiling on device: {device}")
    
    env_name = "square_low_alias"
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
    
    # Mock labels for GT
    gt = {"betti_0": 1, "betti_1": 0}
    tracker = ConvergenceTracker(env, model, gt, cfg)
    
    timings = {
        "rollout": [],
        "forward": [],
        "backward": [],
        "optimizer": [],
        "eval_geom": [],
        "eval_topo": []
    }
    
    for trial in range(1, cfg["n_trials"] + 1):
        # 1. Rollout
        start = time.time()
        rollout = env.sample_rollout(cfg["seq_duration"], seed=trial)
        obs_t = torch.as_tensor(rollout.observations[None, ...], dtype=torch.float32, device=device)
        actions = act_enc.encode(rollout.actions, rollout.headings)
        act_t = torch.as_tensor(actions[None, ...], dtype=torch.float32, device=device)
        timings["rollout"].append(time.time() - start)
        
        # 2. Forward
        start = time.time()
        model.train()
        outputs = model.forward_sequence(obs_t, act_t, training=True)
        pred = outputs["decoded_predictions"]
        target = outputs["raw_targets"]
        hidden = outputs["hidden"]
        timings["forward"].append(time.time() - start)
        
        # 3. Backward
        start = time.time()
        loss = loss_fn(pred, target, hidden=hidden)
        model.zero_grad()
        loss.backward()
        timings["backward"].append(time.time() - start)
        
        # 4. Optimizer
        start = time.time()
        model.optimizer_step(gradient_clip=1.0)
        timings["optimizer"].append(time.time() - start)
        
        # 5. Evaluation
        if trial % cfg["eval_every_trials"] == 0:
            print(f"Trial {trial}: Running geometric evaluation...")
            start = time.time()
            tracker.evaluate(trial, float(loss.detach()))
            timings["eval_geom"].append(time.time() - start)
            
        if trial % cfg["topo_eval_every_trials"] == 0:
            print(f"Trial {trial}: Running topological evaluation...")
            start = time.time()
            tracker.evaluate_topology(trial)
            timings["eval_topo"].append(time.time() - start)

    print("\n--- Profiling Results (Averages) ---")
    for key, values in timings.items():
        if values:
            avg = np.mean(values)
            tot = np.sum(values)
            print(f"{key:10}: Avg: {avg:7.4f}s, Total: {tot:7.2f}s (n={len(values)})")

if __name__ == "__main__":
    profile_training()
