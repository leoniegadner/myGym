import argparse
import json
import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from mbrl.types import TransitionBatch


class MLPPolicy(nn.Module):
    """Simple MLP policy used to mimic PETS planned actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            input_dim = hidden
        layers.append(nn.Linear(input_dim, act_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PetsPolicyMimic:
    """
    Trains a feed-forward policy to imitate PETS actions using supervised learning.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        lr: float = 1e-3,
        val_split: float = 0.1,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = MLPPolicy(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.val_split = max(0.0, min(0.5, float(val_split)))
        self.obs_mean: Optional[torch.Tensor] = None
        self.obs_std: Optional[torch.Tensor] = None
        self.last_stats: Dict[str, float] = {}

    @staticmethod
    def _flatten_obs(obs: np.ndarray) -> np.ndarray:
        if obs.ndim > 2:
            return obs.reshape(obs.shape[0], -1)
        return obs

    def _prepare_data(
        self, replay_buffer
    ) -> Optional[Tuple[TensorDataset, int, int]]:
        if replay_buffer is None or len(replay_buffer) == 0:
            return None

        transitions = replay_buffer.get_all(shuffle=False)
        obs = np.asarray(transitions.obs, dtype=np.float32)
        actions = np.asarray(transitions.act, dtype=np.float32)

        obs = self._flatten_obs(obs)
        if actions.ndim > 2:
            actions = actions.reshape(actions.shape[0], -1)

        if len(obs) == 0:
            return None

        print(f"[mimic] training samples: {len(obs)}")

        obs_t = torch.as_tensor(obs, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)

        self.obs_mean = obs_t.mean(dim=0, keepdim=True)
        self.obs_std = obs_t.std(dim=0, keepdim=True).clamp(min=1e-6)

        dataset = TensorDataset(obs_t, actions_t)
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size

        return dataset, train_size, val_size

    def train_from_replay(
        self,
        replay_buffer,
        batch_size: int = 512,
        epochs: int = 10,
    ) -> Dict[str, float]:
        prep = self._prepare_data(replay_buffer)
        if prep is None:
            self.last_stats = {"train_loss": float("nan"), "val_loss": float("nan")}
            return self.last_stats

        dataset, train_size, val_size = prep
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_set, batch_size=min(batch_size, max(train_size, 1)), shuffle=True
        )
        val_loader: Optional[DataLoader] = None
        if val_size > 0:
            val_loader = DataLoader(
                val_set, batch_size=min(batch_size, val_size), shuffle=False
            )

        loss_fn = nn.MSELoss()
        for epoch_idx in range(max(epochs, 1)):
            self.model.train()
            train_loss = 0.0
            total = 0
            train_within_tol = 0
            train_dist_sum = 0.0
            for obs_batch, action_batch in train_loader:
                norm_obs = (obs_batch - self.obs_mean) / self.obs_std
                pred = self.model(norm_obs)
                loss = loss_fn(pred, action_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * len(obs_batch)
                total += len(obs_batch)
                train_within_tol += (torch.norm(pred - action_batch, dim=-1) < 0.05).sum().item()
                train_dist_sum += torch.norm(pred - action_batch, dim=-1).sum().item()
            mean_train_loss = train_loss / max(total, 1)
            train_success_rate = train_within_tol / max(total, 1)
            train_mean_distance = train_dist_sum / max(total, 1)

            mean_val_loss = float("nan")
            val_success_rate = float("nan")
            val_mean_distance = float("nan")
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                total_val = 0
                val_within_tol = 0
                val_dist_sum = 0.0
                with torch.no_grad():
                    for obs_batch, action_batch in val_loader:
                        norm_obs = (obs_batch - self.obs_mean) / self.obs_std
                        pred = self.model(norm_obs)
                        loss = loss_fn(pred, action_batch)
                        val_loss += loss.item() * len(obs_batch)
                        total_val += len(obs_batch)
                        val_within_tol += (torch.norm(pred - action_batch, dim=-1) < 0.05).sum().item()
                        val_dist_sum += torch.norm(pred - action_batch, dim=-1).sum().item()
                mean_val_loss = val_loss / max(total_val, 1)
                val_success_rate = val_within_tol / max(total_val, 1)
                val_mean_distance = val_dist_sum / max(total_val, 1)

            self.last_stats = {
                "train_loss": float(mean_train_loss),
                "val_loss": float(mean_val_loss),
                "train_success_rate": float(train_success_rate),
                "val_success_rate": float(val_success_rate),
                "train_mean_distance": float(train_mean_distance),
                "val_mean_distance": float(val_mean_distance),
            }
            print(
                f"[mimic][epoch {epoch_idx+1}/{epochs}] "
                f"train_loss={mean_train_loss:.6f} "
                f"val_loss={mean_val_loss:.6f} "
                f"train_success={train_success_rate:.4f} "
                f"val_success={val_success_rate:.4f} "
                f"train_dist={train_mean_distance:.4f} "
                f"val_dist={val_mean_distance:.4f}"
            )

        return self.last_stats

    def predict(self, obs: np.ndarray) -> np.ndarray:
        self.model.eval()
        obs_np = np.asarray(obs, dtype=np.float32)
        obs_np = self._flatten_obs(obs_np)
        obs_t = torch.as_tensor(obs_np, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        if self.obs_mean is not None and self.obs_std is not None:
            obs_t = (obs_t - self.obs_mean) / self.obs_std
        with torch.no_grad():
            action = self.model(obs_t)
        return action.cpu().numpy()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "obs_mean": None if self.obs_mean is None else self.obs_mean.cpu(),
                "obs_std": None if self.obs_std is None else self.obs_std.cpu(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("obs_mean") is not None:
            self.obs_mean = checkpoint["obs_mean"].to(self.device)
        if checkpoint.get("obs_std") is not None:
            self.obs_std = checkpoint["obs_std"].to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Iterable[int],
        device: str = "cpu",
    ) -> "PetsPolicyMimic":
        instance = cls(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        )
        instance.load(path)
        return instance


class _NPZReplayAdapter:
    """Minimal adapter to feed replay_buffer.npz data into the mimic trainer."""

    def __init__(self, data: Dict[str, np.ndarray]):
        required = ["obs", "next_obs", "action", "reward", "terminated", "truncated"]
        for key in required:
            if key not in data:
                raise ValueError(f"replay_buffer.npz missing required array '{key}'")
        self.data = data
        self.trajectory_indices = data.get("trajectory_indices", [])

    def __len__(self):
        return len(self.data["obs"])

    def get_all(self, shuffle: bool = False) -> TransitionBatch:
        idx = np.arange(len(self))
        if shuffle:
            idx = np.random.permutation(idx)
        return TransitionBatch(
            self.data["obs"][idx],
            self.data["action"][idx],
            self.data["next_obs"][idx],
            self.data["reward"][idx],
            self.data["terminated"][idx],
            self.data["truncated"][idx],
        )


def _parse_hidden_sizes(hidden: str) -> Sequence[int]:
    if isinstance(hidden, (list, tuple)):
        return [int(h) for h in hidden]
    try:
        return [int(x.strip()) for x in str(hidden).split(",") if x.strip()]
    except Exception:
        return [256, 256]


def train_mimic_from_dir(
    model_dir: str,
    hidden_sizes: Sequence[int],
    batch_size: int = 512,
    epochs: int = 10,
    lr: float = 1e-3,
    val_split: float = 0.1,
    device: str = "cpu",
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """Train a policy mimic from a saved replay_buffer.npz after model training."""
    buffer_path = os.path.join(model_dir, "replay_buffer.npz")
    if not os.path.isfile(buffer_path):
        raise FileNotFoundError(f"No replay_buffer.npz found in {model_dir}")

    data = np.load(buffer_path)
    adapter = _NPZReplayAdapter(data)
    obs_dim = int(np.prod(data["obs"].shape[1:]))
    act_dim = int(np.prod(data["action"].shape[1:]))

    mimic = PetsPolicyMimic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden_sizes,
        lr=lr,
        val_split=val_split,
        device=device,
    )
    stats = mimic.train_from_replay(adapter, batch_size=batch_size, epochs=epochs)
    save_path = output_path or os.path.join(model_dir, "pets_policy_mimic.pth")
    mimic.save(save_path)

    stats_path = os.path.join(
        model_dir, "pets_policy_mimic_stats.json"
    ) if output_path is None else os.path.splitext(save_path)[0] + "_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[mimic] saved policy to {save_path}")
    print(f"[mimic] stats: {stats}")
    return stats


def _build_argparser():
    parser = argparse.ArgumentParser(description="Train PETS policy mimic from a saved replay_buffer.npz")
    parser.add_argument("--model_dir", required=True, help="Directory containing replay_buffer.npz and train.json")
    parser.add_argument("--hidden_sizes", default="256,256", help="Comma-separated hidden sizes, e.g. 256,256")
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split (0-0.5)")
    parser.add_argument("--device", default="cpu", help="torch device for training")
    parser.add_argument("--output_path", default=None, help="Optional override path for the saved mimic (.pth)")
    return parser


def main():
    parser = _build_argparser()
    args = parser.parse_args()
    hidden_sizes = _parse_hidden_sizes(args.hidden_sizes)
    train_mimic_from_dir(
        model_dir=args.model_dir,
        hidden_sizes=hidden_sizes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        device=args.device,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
