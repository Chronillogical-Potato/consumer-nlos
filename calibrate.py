import matplotlib.pyplot as plt
import numpy as np

from config import Config
from sensor import (
    average_histograms,
    average_point_cloud,
    build_sensor,
    capture_frames,
)


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Align a set of 3D points to z=0 via rotation then translation.

    Returns the aligned points and the camera height |z_0|.
    """
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid

    _, _, vt = np.linalg.svd(centered)
    normal = vt[-1] / np.linalg.norm(vt[-1])
    if normal[2] > 0:
        normal = -normal

    phi = -np.arcsin(normal[1])
    theta = np.pi - np.arctan2(normal[0], normal[2])

    r_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    r_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)],
    ])
    rot = r_x @ r_y

    aligned = (rot @ centered.T).T
    cam_trans = rot @ (-centroid.reshape(3))
    aligned[:, 0] -= cam_trans[0]
    aligned[:, 1] -= cam_trans[1]

    return aligned, float(np.abs(cam_trans[2]))


def compute_calibration(
    frames: list[dict],
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    pt_cloud = average_point_cloud(frames)
    pt_cloud, cam_z = fit_plane(pt_cloud)

    # swap x/y so +x points right (matches the original convention)
    x_old = pt_cloud[:, 1].copy()
    y_old = pt_cloud[:, 0].copy()
    pt_cloud[:, 0] = -x_old
    pt_cloud[:, 1] = y_old

    hists = average_histograms(frames)
    bin_0 = np.argmax(hists, axis=-1).reshape(-1)

    return pt_cloud, cam_z, bin_0, hists


def save_calibration(
    out_dir, pt_cloud: np.ndarray, cam_z: float, bin_0: np.ndarray
) -> None:
    out_dir.mkdir(exist_ok=True, parents=True)
    np.savez(
        out_dir / "calibration.npz",
        pt_cloud=pt_cloud,
        cam_z=cam_z,
        bin_0=bin_0,
    )


def plot_diagnostics(
    out_dir,
    pt_cloud: np.ndarray,
    cam_z: float,
    hists: np.ndarray,
    bin_0: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(15, 5))
    plt.title(f"Point Cloud (camera z = {cam_z:.2f} m)")
    for ax in fig.axes:
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(
            axis="both", which="both",
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labelleft=False,
        )
    plt.subplot(1, 3, 1); plt.scatter(pt_cloud[:, 0], pt_cloud[:, 1])
    plt.subplot(1, 3, 2); plt.scatter(pt_cloud[:, 0], pt_cloud[:, 2])
    plt.subplot(1, 3, 3); plt.scatter(pt_cloud[:, 1], pt_cloud[:, 2])
    plt.savefig(out_dir / "pt_cloud.png")
    plt.close()

    plt.figure()
    flat = hists.reshape(-1, hists.shape[-1])
    plt.imshow(flat, cmap="hot")
    plt.plot(bin_0, np.arange(bin_0.shape[0]), "g--")
    plt.savefig(out_dir / "histogram.png")
    plt.close()


def main(cfg: Config = Config()) -> None:
    sensor = build_sensor(cfg, start_bin=cfg.start_bin_calibrate)
    try:
        frames = capture_frames(
            sensor,
            n=cfg.ranging_frequency_hz * int(cfg.background_seconds),
            desc="Calibrating point cloud data",
        )
        pt_cloud, cam_z, bin_0, hists = compute_calibration(frames)
        save_calibration(cfg.calibration_dir, pt_cloud, cam_z, bin_0)
        plot_diagnostics(cfg.calibration_dir, pt_cloud, cam_z, hists, bin_0)
        input(
            f"Point cloud saved to {cfg.calibration_dir}. "
            "ctrl-c to close program..."
        )
    except KeyboardInterrupt:
        pass
    finally:
        sensor.close()


if __name__ == "__main__":
    main()
