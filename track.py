import time

import numpy as np

from config import Config
from dashboard import Dashboard
from particle_filter import ParticleFilter
from sensor import average_histograms, build_sensor, capture_frames


def load_calibration(calibration_dir) -> tuple[np.ndarray, float, np.ndarray]:
    data = np.load(calibration_dir / "calibration.npz")
    return data["pt_cloud"], float(data["cam_z"]), data["bin_0"]


def capture_background(
    sensor, cfg: Config, bin_0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Capture stationary frames to build a background histogram and 1B mask.

    Saves to ``cfg.background_dir/calibration.npz`` so the calibration is
    reusable across runs.
    """
    background_file = cfg.background_dir / "calibration.npz"

    input("Press Enter to calibrate background...")
    time.sleep(cfg.pre_capture_pause_s)

    n = cfg.ranging_frequency_hz * int(cfg.background_seconds)
    frames = capture_frames(sensor, n=n, desc="Accumulating background data")

    hists = average_histograms(frames)
    background = hists.reshape(-1, hists.shape[-1])  # (num_pixels, num_bins)

    # mask out the 1B peak region per pixel
    hist_mask = np.ones_like(background)
    start_frame = sensor.start_bin
    for i in range(hist_mask.shape[0]):
        cutoff = bin_0[i] + cfg.pulse_half_width - start_frame - 1
        if bin_0[i] + cfg.pulse_half_width > start_frame - 1:
            hist_mask[i, :cutoff] = 0

    if not background_file.exists():
        cfg.background_dir.mkdir(exist_ok=True, parents=True)
        np.savez(background_file, hist_mask=hist_mask, background=background)

    return background, hist_mask


def preprocess_hist(
    frame: dict,
    background: np.ndarray,
    hist_mask: np.ndarray,
    bin_0: np.ndarray,
    start_frame: int,
    num_lct_bins: int,
    zero_first_k_bins: int,
) -> np.ndarray:
    """Background-subtract, mask 1B, and shift each pixel so the 1B peak is at bin 0."""
    num_pixels = bin_0.shape[0]
    hists = frame["histogram"].reshape(num_pixels, -1)
    num_bins = hists.shape[1]

    hists = np.maximum(hists - background, 0)
    hists *= hist_mask

    shifted = np.zeros((num_pixels, num_lct_bins), dtype=hists.dtype)
    for i in range(num_pixels):
        if start_frame - 1 < bin_0[i]:
            cropped = hists[i, bin_0[i] - (start_frame - 1) :]
            shifted[i, : cropped.shape[0]] = cropped
        else:
            pad = (start_frame - 1) - bin_0[i]
            shifted[i, pad : pad + num_bins] = hists[i, :]

    shifted[:, :zero_first_k_bins] = 0
    return shifted


def main(cfg: Config = Config()) -> None:
    pt_cloud, cam_z, bin_0 = load_calibration(cfg.calibration_dir)

    sensor = build_sensor(cfg, start_bin=cfg.start_bin_track)
    dashboard = None
    try:
        background, hist_mask = capture_background(sensor, cfg, bin_0)

        input("Press Enter to start data capture...")
        print("Starting data capture...")
        time.sleep(cfg.pre_capture_pause_s)

        pf = ParticleFilter(cfg, t_res=sensor.timing_resolution)
        dashboard = Dashboard(cfg, cam_z=cam_z, z_range=cfg.z_range)

        frame_count = 0
        t0 = time.time()
        while dashboard.is_okay:
            frame = sensor.accumulate()
            hist = preprocess_hist(
                frame,
                background,
                hist_mask,
                bin_0,
                start_frame=cfg.start_bin_track,
                num_lct_bins=cfg.num_lct_bins,
                zero_first_k_bins=cfg.zero_first_k_bins,
            )
            particles = pf.update(pt_cloud, hist)
            dashboard.update(particles, hist[:, : hist.shape[1]], pt_cloud)

            frame_count += 1
            if frame_count % 10 == 0:
                t1 = time.time()
                print(f"Frame: {frame_count}, FPS: {10 / (t1 - t0):.2f}")
                t0 = t1
    except KeyboardInterrupt:
        pass
    finally:
        if dashboard is not None:
            try:
                dashboard.close()
            except Exception:
                pass
        sensor.close()


if __name__ == "__main__":
    main()
