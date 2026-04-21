from utils.Architectures import pRNN_th
from utils.thetaRNN import LayerNormRNNCell
from project5_symmetry.training.train import _build_optimizer
import torch
import torch.nn.functional as F

model = pRNN_th(
    obs_size=147, act_size=5, k=5, hidden_size=500,
    cell=LayerNormRNNCell, dropp=0.15, neuralTimescale=2,
)
optimizer = _build_optimizer(model)
obs = torch.rand(1, 201, 147)
act = torch.zeros(1, 200, 5)
act[:, :, 0] = 1.0

loss_step0 = None
for step in range(4):
    pred, h, target = model(obs, act)
    loss = F.mse_loss(pred, target)
    if step == 0:
        loss_step0 = loss.item()

        # Check A — pred std across rollout steps
        print("=== Check A: Pred std per rollout step ===")
        for i in range(6):
            print(f"  step {i}: {pred[0,i].std().item():.6f}")

        # Within-timestep correlation at t=97
        t = 97
        print(f"\n=== Within-timestep correlations at t={t} ===")
        for i in range(6):
            for j in range(i+1, 6):
                corr = torch.corrcoef(torch.stack([
                    pred[0, i, t, :],
                    pred[0, j, t, :]
                ]))[0,1]
                print(f"  corr(step_{i}, step_{j}): {corr:.4f}")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

print(f"\n=== Check B: Loss ===")
print(f"step 0 loss: {loss_step0:.6f}")
print(f"step 3 loss: {loss.item():.6f}")
