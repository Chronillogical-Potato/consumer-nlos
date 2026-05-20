import numpy as np
import tqdm

from config import Config
from spad_driver import RangingMode, VL53L8CHSensor


def build_sensor(cfg: Config, start_bin: int) -> VL53L8CHSensor:
    """Instantiate the VL53L8CH driver from a `Config`."""
    return VL53L8CHSensor(
        height=4,
        width=4,
        num_bins=cfg.num_bins,
        start_bin=start_bin,
        subsample=cfg.subsample,
        ranging_mode=RangingMode.CONTINUOUS,
        ranging_frequency_hz=cfg.ranging_frequency_hz,
    )


def capture_frames(sensor: VL53L8CHSensor, n: int, desc: str) -> list[dict]:
    return [
        sensor.accumulate()
        for _ in tqdm.tqdm(range(n), leave=False, desc=desc)
    ]


def average_point_cloud(frames: list[dict]) -> np.ndarray:
    return np.nanmean([f["point_cloud"] for f in frames], axis=0)


def average_histograms(frames: list[dict]) -> np.ndarray:
    return np.mean([f["histogram"] for f in frames], axis=0)
