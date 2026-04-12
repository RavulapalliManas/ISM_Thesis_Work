
import time
from project4_topology_before_geometry.environments.env_factory import get_env
from project4_topology_before_geometry.sensory.aliasing_control import compute_geo_euclidean_discrepancy, compute_aliasing_score

def test_speed():
    cfg = {
        "hidden_dim": 256,
        "rollout_k": 5,
        "seq_duration": 100,
        "n_trials": 5000,
        "eval_every_trials": 100,
        "topo_eval_every_trials": 500,
        "checkpoint_every_trials": 1000,
        "loss_type": "rollout_mse",
        "recurrence_scale": 1.0,
        "sigma_noise": 0.03,
        "dropout": 0.15,
        "neural_timescale": 2,
        "time_mode": "continuous",
        "device": "cpu",
        "topo_device": "cpu",
        "mixed_precision": False,
        "use_wandb": False,
        "seed": 42,
        "gradient_clip": 1.0,
        "environments": ["square_low_alias"],
        "env_backend": {"square_low_alias": "minigrid"}
    }
    
    env_name = "square_low_alias"
    print(f"Testing environment: {env_name}")
    
    start = time.time()
    env = get_env(env_name, cfg)
    print(f"Env creation took: {time.time() - start:.2f}s")
    
    start = time.time()
    score = compute_aliasing_score(env)
    print(f"compute_aliasing_score took: {time.time() - start:.2f}s (score: {score})")
    
    start = time.time()
    discrepancy = compute_geo_euclidean_discrepancy(env)
    print(f"compute_geo_euclidean_discrepancy took: {time.time() - start:.2f}s")

if __name__ == "__main__":
    test_speed()
