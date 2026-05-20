from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # --- Sensor ---
    num_bins: int = 48
    start_bin_calibrate: int = 1
    start_bin_track: int = 30
    ranging_frequency_hz: int = 30
    subsample: int = 1

    # --- Paths ---
    log_root: Path = Path("logs")
    calibration_dir: Path = Path("logs/pt_cloud")
    background_dir: Path = Path("logs/background")
    canon_path: Path = Path(__file__).resolve().parent / "canons" / "point.npy"

    # --- Background / 1B removal ---
    background_seconds: float = 2.0
    pulse_half_width: int = 7
    zero_first_k_bins: int = 15
    pre_capture_pause_s: float = 5.0

    # --- Particle filter ---
    num_particles: int = 1000
    x_range: tuple[float, float] = (0.0, 1.0)
    y_range: tuple[float, float] = (-0.6, 0.6)
    z_range: tuple[float, float] = (0.0, 3.0)
    radius: float = 0.07
    eta: float = 3.0
    min_z: float = 0.1

    # --- Forward model (canonical) ---
    canon_num_x: int = 400
    canon_num_y: int = 400
    canon_x_min: float = -4.0
    canon_x_max: float = 4.0
    canon_y_min: float = -4.0
    canon_y_max: float = 4.0
    num_lct_bins: int = 128
    num_sub_bins: int = 10

    # --- Dashboard ---
    dashboard_xlim: tuple[float, float] = (-0.3, 1.5)
    occluder_x: float = 0.3
    occluder_wall_length: float = 0.7
    trail_length: int = 10
    fullscreen: bool = True
