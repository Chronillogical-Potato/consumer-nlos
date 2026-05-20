import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from tqdm import tqdm

from config import Config

SPEED_OF_LIGHT = 3e8


class FastSumOfParabolas(nn.Module):
    """Pre-voxelized sum-of-parabolas forward model for a single canonical object."""

    def __init__(self, cfg: Config, t_res: float, device: str) -> None:
        super().__init__()

        self.device = device
        self.num_x = cfg.canon_num_x
        self.num_y = cfg.canon_num_y
        self.x_min = cfg.canon_x_min
        self.x_max = cfg.canon_x_max
        self.y_min = cfg.canon_y_min
        self.y_max = cfg.canon_y_max
        self.num_bins = cfg.num_lct_bins
        self.num_sub_bins = cfg.num_sub_bins

        self.v_range = (SPEED_OF_LIGHT * self.num_bins * t_res / 2) ** 2
        self.num_v = self.num_sub_bins * self.num_bins
        self.v_base_res = self.v_range / self.num_v
        self.pad_length = self.num_v

        points = np.load(cfg.canon_path)
        canon_voxel = self._voxelize(points)
        self.canon_voxel = torch.from_numpy(canon_voxel).to(device, non_blocking=True)

    def _voxelize(self, points: np.ndarray) -> np.ndarray:
        canon = np.zeros(
            (self.num_y, self.num_x, self.num_v + 2 * self.pad_length),
            dtype=np.float32,
        )
        canon_x = np.linspace(self.x_min, self.x_max, self.num_x)
        canon_y = np.linspace(self.y_min, self.y_max, self.num_y)
        ones_vec = np.ones_like(canon_y)
        num_points = points.shape[0]

        for i in tqdm(range(self.num_y), desc="Voxelizing canonical measurement"):
            query = np.stack([canon_x, ones_vec * canon_y[i]], axis=-1)
            v_loc = np.sum(
                (query[None, :, :] - points[:, None, 0:2]) ** 2, axis=-1
            )
            v_loc += points[:, 2].reshape(-1, 1) ** 2
            v_idx = np.floor(v_loc / self.v_base_res).astype(int) + self.pad_length

            x_idx = np.tile(np.arange(self.num_x).reshape(1, -1), (num_points, 1))
            y_idx = np.full((num_points, self.num_x), i, dtype=int)
            x_flat = x_idx.reshape(-1)
            y_flat = y_idx.reshape(-1)
            v_flat = v_idx.reshape(-1)
            weights = np.ones_like(x_flat)

            for k in range(self.num_sub_bins):
                offset = k - self.num_sub_bins // 2
                np.add.at(canon, (y_flat, x_flat, v_flat + offset), weights)

        return canon

    def forward(self, pt_cloud: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Render histograms for a batch of particles.

        pt_cloud : (num_pixels, 3) wall sample locations.
        deltas   : (batch, 3) per-particle (x, y, v) shifts.

        Returns  : (batch, num_pixels, num_bins).
        """
        num_pixels = pt_cloud.shape[0]
        batch = deltas.shape[0]

        x_samp = pt_cloud[:, 0].unsqueeze(0).repeat(batch, 1) - deltas[:, 0:1]
        y_samp = pt_cloud[:, 1].unsqueeze(0).repeat(batch, 1) - deltas[:, 1:2]

        x_check = x_samp.detach().to("cpu") if self.device == "mps" else x_samp
        y_check = y_samp.detach().to("cpu") if self.device == "mps" else y_samp
        if (x_check < self.x_min).any() or (x_check > self.x_max).any():
            raise ValueError(
                f"x_samp out of bounds: {x_check.min().item()} {x_check.max().item()}"
            )
        if (y_check < self.y_min).any() or (y_check > self.y_max).any():
            raise ValueError(
                f"y_samp out of bounds: {y_check.min().item()} {y_check.max().item()}"
            )

        x_idx = (self.num_x * (x_samp - self.x_min) / (self.x_max - self.x_min))
        y_idx = (self.num_y * (y_samp - self.y_min) / (self.y_max - self.y_min))
        x_idx = x_idx.unsqueeze(-1).repeat(1, 1, self.num_bins).int().reshape(-1)
        y_idx = y_idx.unsqueeze(-1).repeat(1, 1, self.num_bins).int().reshape(-1)

        v_axis = torch.linspace(0, self.v_range, self.num_bins).to(self.device)
        v_samp = -deltas[:, 2].reshape(batch, 1, 1) + v_axis.reshape(1, 1, -1)
        v_samp = v_samp.expand(batch, num_pixels, self.num_bins)
        v_idx = (self.pad_length + (v_samp / self.v_base_res).int()).reshape(-1)

        hists = self.canon_voxel[y_idx, x_idx, v_idx]
        return hists.reshape(batch, num_pixels, self.num_bins)


def dot_product_score(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return torch.sum(gt * pred, dim=tuple(range(1, gt.dim())))


def residual_resample(scores: torch.Tensor) -> torch.Tensor:
    """Residual resampling — deterministic integer copies + multinomial remainder."""
    n = scores.shape[0]
    probs = (scores / scores.sum()).cpu()
    probs[torch.isnan(probs)] = 0

    num_copies = np.floor(probs.numpy() * n).astype(int)
    residual = probs * n - num_copies
    residual[torch.isnan(residual) | (residual < 0)] = 0
    residual /= residual.sum()

    indices: list[int] = []
    for i in range(n):
        if num_copies[i] != 0:
            indices.extend([i] * num_copies[i])

    remaining = n - len(indices)
    if remaining > 0:
        indices.extend(
            np.random.choice(range(n), size=remaining, p=residual.numpy())
        )

    return torch.tensor(indices, dtype=torch.long, device=scores.device)


def resampling_operator(num_bins: int):
    """Build the LCT resampling matrix (O'Toole et al., Confocal NLOS Imaging).

    Returns (mtx, mtxi) where mtx maps a native-resolution histogram into the
    squared-time (v) space that the canonical model lives in.
    """
    mtx = csr_matrix(([], ([], [])), shape=(num_bins ** 2, num_bins))
    x = np.arange(1, num_bins ** 2 + 1)
    mtx[x - 1, np.ceil(np.sqrt(x)).astype(int) - 1] = 1

    diag = csr_matrix(
        (1 / np.sqrt(x), (np.arange(num_bins ** 2), np.arange(num_bins ** 2))),
        shape=(num_bins ** 2, num_bins ** 2),
    )
    mtx = diag.dot(mtx)
    mtxi = mtx.transpose()

    k_levels = int(np.round(np.log(num_bins) / np.log(2)))
    for _ in range(k_levels):
        mtx = 0.5 * (mtx[0::2, :] + mtx[1::2, :])
        mtxi = 0.5 * (mtxi[:, 0::2] + mtxi[:, 1::2])

    return mtx, mtxi


class ParticleFilter:
    def __init__(self, cfg: Config, t_res: float):
        self.cfg = cfg
        # The original code forced CPU even when MPS was available; preserve.
        self.device = "cpu"

        self.canon = FastSumOfParabolas(cfg, t_res=t_res, device=self.device)
        self.eta = cfg.eta
        self.radius = cfg.radius
        self.min_z = cfg.min_z

        self.particles = self._init_particles()
        self.velocity = self._init_velocity()
        self.mtx, _ = resampling_operator(cfg.num_lct_bins)

    def _init_particles(self) -> torch.Tensor:
        cfg = self.cfg
        n = cfg.num_particles
        x = torch.rand(n) * (cfg.x_range[1] - cfg.x_range[0]) + cfg.x_range[0]
        y = torch.rand(n) * (cfg.y_range[1] - cfg.y_range[0]) + cfg.y_range[0]
        z = torch.rand(n) * (cfg.z_range[1] - cfg.z_range[0]) + cfg.z_range[0]
        return torch.stack([x, y, z], dim=-1).to(self.device)

    def _init_velocity(self) -> torch.Tensor:
        n = self.cfg.num_particles
        r = self.radius * torch.pow(torch.rand(n), 1 / 3)
        phi = 2 * torch.pi * torch.rand(n)
        theta = torch.pi * torch.rand(n)
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        return torch.stack([x, y, z], dim=-1).to(self.device)

    def update(self, pt_cloud_np: np.ndarray, hist_np: np.ndarray) -> np.ndarray:
        hist_lct = (self.mtx @ hist_np.T).T  # (num_pixels, num_lct_bins)
        pt_cloud = torch.from_numpy(pt_cloud_np.astype(np.float32)).to(self.device)
        hist = torch.from_numpy(hist_lct.astype(np.float32)).to(self.device)

        scores = self._score(pt_cloud, hist)
        self._resample(scores ** self.eta)
        current = self.particles.detach().cpu().numpy().copy()
        self._propagate()
        return current

    def _score(self, pt_cloud: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        batch = self.particles.clone()
        # convert z to v-space (signed square)
        batch[:, 2] = torch.sign(batch[:, 2]) * batch[:, 2] ** 2

        with torch.no_grad():
            y_hat = self.canon(pt_cloud=pt_cloud, deltas=batch)

        scores = dot_product_score(
            hist.unsqueeze(0).expand(self.cfg.num_particles, -1, -1), y_hat
        )

        # zero out particles too close to the wall
        scores[self.particles[:, 2] < self.min_z] = 0

        # ensure non-negative, replace NaNs with 0
        nan_mask = torch.isnan(scores)
        valid = scores[~nan_mask]
        if valid.numel():
            scores = scores - valid.min()
        scores[nan_mask] = 0
        return scores

    def _resample(self, scores: torch.Tensor) -> None:
        idx = residual_resample(scores)
        self.particles = self.particles[idx]
        self.velocity = self.velocity[idx]

    def _propagate(self) -> None:
        # random walk: zero-mean Gaussian, std = radius
        self.velocity = torch.randn_like(self.velocity) * self.radius
        self.particles = self.particles + self.velocity
        # restrict z to positive half-space
        self.particles[:, 2] = self.particles[:, 2].clamp(min=0)
