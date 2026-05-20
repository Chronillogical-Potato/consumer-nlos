"""Self-contained driver for the VL53L8CH time-of-flight SPAD sensor.

Vendored and trimmed from cc-hardware (https://github.com/camera-culture/cc-hardware):
  cc_hardware/drivers/spads/vl53l8ch.py
  cc_hardware/drivers/spads/spad.py
  cc_hardware/drivers/safe_serial.py
  cc_hardware/utils/misc/serial_utils.py

Removed: framework scaffolding (Sensor/Config/Component/Registry/hydra_config
base classes, Settings, SafeSerial lock, SPADDataType Flag enum, logger,
flash-script path). The string keys 'histogram', 'point_cloud', and 'distance'
replace SPADDataType in the returned frame dict.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.queues
import multiprocessing.synchronize
import struct
import time
from enum import Enum

import numpy as np
import serial
from serial.serialutil import SerialException
from serial.tools import list_ports

SPEED_OF_LIGHT = 299_792_458.0  # m/s
BAUDRATE = 2_250_000


class RangingMode(Enum):
    CONTINUOUS = 1
    AUTONOMOUS = 3


def find_first_port() -> str:
    """Return the device path of the first serial port that looks like a USB device."""
    candidates = []
    for port in list_ports.comports():
        if port.pid is None:
            continue
        try:
            # Quick open/close to confirm the port is accessible.
            serial.Serial(port.device).close()
        except SerialException:
            continue
        candidates.append(port.device)

    if not candidates:
        raise RuntimeError("No serial ports found for VL53L8CH sensor")
    return candidates[0]


def _pack_config(
    *,
    height: int,
    width: int,
    ranging_mode: RangingMode,
    ranging_frequency_hz: int,
    integration_time_ms: int,
    start_bin: int,
    num_bins: int,
    subsample: int,
    agg_start_x: int,
    agg_start_y: int,
    agg_merge_x: int,
    agg_merge_y: int,
) -> bytes:
    """Pack the 13-uint16 sensor configuration that the firmware expects."""
    return struct.pack(
        "<13H",
        height * width,
        ranging_mode.value,
        ranging_frequency_hz,
        integration_time_ms,
        start_bin,
        num_bins,
        subsample,
        agg_start_x,
        agg_start_y,
        agg_merge_x,
        agg_merge_y,
        width,
        height,
    )


class _FrameAccumulator:
    """Builds one histogram frame from per-pixel rows emitted by the sensor."""

    def __init__(self, height: int, width: int, num_bins: int, add_back_ambient: bool):
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.num_pixels = height * width
        self.add_back_ambient = add_back_ambient
        self.reset()

    def reset(self) -> None:
        self._histogram: dict[int, np.ndarray] = {}
        self._distance: dict[int, float] = {}
        self._done = False

    @property
    def is_done(self) -> bool:
        return self._done

    def process_row(self, row: list[str]) -> bool:
        """Parse one row. Returns False on a parse error (caller should reset).

        Rows whose bin count doesn't match ``self.num_bins`` are rejected so
        that stale frames from a previous sensor configuration don't slip into
        the accumulator and blow up on the final reshape.
        """
        if len(row) - 3 != self.num_bins:
            return False
        try:
            idx = int(row[0])
            if idx in self._histogram:
                return False
            ambient = int(row[1]) if self.add_back_ambient else 0
            bins = np.array([int(v) + ambient for v in row[3:]])
            self._histogram[idx] = np.clip(bins, 0, None)
            self._distance[idx] = float(row[2])
        except (ValueError, IndexError):
            return False

        if len(self._histogram) == self.num_pixels:
            self._done = True
        return True

    def histogram(self) -> np.ndarray:
        flat = np.array([self._histogram[i] for i in sorted(self._histogram)])
        return flat.reshape(self.height, self.width, self.num_bins)

    def distance(self) -> np.ndarray:
        flat = np.array([self._distance[i] for i in sorted(self._distance)])
        return flat.reshape(self.height, self.width)


def _compute_distance(
    histogram: np.ndarray,
    *,
    timing_resolution: float,
    subsample: int,
    start_bin: int,
    window: int = 10,
    threshold: float = 0.0,
) -> np.ndarray:
    """Weighted-centroid time-of-flight distance per pixel (millimetres).

    Mirrors cc_hardware/drivers/spads/spad.py::calculate_distance.
    """
    h, w, num_bins = histogram.shape
    out = np.zeros((h, w), dtype=float)
    for i in range(h):
        for j in range(w):
            peak = int(histogram[i, j, :].argmax())
            start = max(0, peak - window // 2)
            end = min(num_bins, peak + window // 2 + 1)
            weights = histogram[i, j, start:end].astype(float)
            if weights.sum() < threshold:
                continue
            weights /= weights.sum()
            bins = np.arange(start, end) + start_bin
            tof = bins * timing_resolution * subsample
            out[i, j] = SPEED_OF_LIGHT * np.dot(weights, tof) / 4 * 1000.0
    return out


def _compute_point_cloud(
    distances: np.ndarray, *, fovx: float, fovy: float
) -> np.ndarray:
    """Back-project per-pixel distances (mm) to a point cloud in metres.

    Mirrors cc_hardware/drivers/spads/spad.py::calculate_point_cloud (with
    subpixel_samples=1, bilinear_interpolation=False).
    """
    h, w = distances.shape
    px_x = np.radians(fovx) / w
    px_y = np.radians(fovy) / h

    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    y_sub = grid_y + 0.5
    x_sub = grid_x + 0.5

    d = np.maximum(0.0, distances[grid_y, grid_x])
    angle_x = y_sub * px_y - np.radians(fovy) / 2 - np.pi / 2
    angle_y = x_sub * px_x - np.radians(fovx) / 2

    x = (d * np.cos(angle_x)) / 1e3
    y = (d * np.sin(angle_y)) / 1e3
    z = d / 1e3
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


class VL53L8CHSensor:
    """Self-contained VL53L8CH SPAD driver.

    Parameters configure the sensor and the data returned by ``accumulate()``.
    The background reader is a daemon ``multiprocessing.Process``; ``close()``
    signals it to stop and joins it (closing the serial port in the process).
    """

    def __init__(
        self,
        *,
        port: str | None = None,
        height: int = 4,
        width: int = 4,
        num_bins: int = 8,
        start_bin: int = 0,
        subsample: int = 1,
        ranging_mode: RangingMode = RangingMode.CONTINUOUS,
        ranging_frequency_hz: int = 30,
        integration_time_ms: int = 10,
        agg_start_x: int = 0,
        agg_start_y: int = 0,
        agg_merge_x: int = 1,
        agg_merge_y: int = 1,
        fovx: float = 45.0,
        fovy: float = 45.0,
        timing_resolution: float = 250e-12,
        add_back_ambient: bool = False,
        serial_wait_s: float = 1.0,
    ) -> None:
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.start_bin = start_bin
        self.subsample = subsample
        self.ranging_mode = ranging_mode
        self.ranging_frequency_hz = ranging_frequency_hz
        self.integration_time_ms = integration_time_ms
        self.fovx = fovx
        self.fovy = fovy
        self.timing_resolution = timing_resolution
        self.add_back_ambient = add_back_ambient
        self.num_pixels = height * width

        self._data = _FrameAccumulator(height, width, num_bins, add_back_ambient)

        self._queue: multiprocessing.Queue = multiprocessing.Queue(
            maxsize=self.num_pixels * 4
        )
        self._write_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=10)
        self._initialized_event = multiprocessing.Event()
        self._stop_event = multiprocessing.Event()

        packed = _pack_config(
            height=height,
            width=width,
            ranging_mode=ranging_mode,
            ranging_frequency_hz=ranging_frequency_hz,
            integration_time_ms=integration_time_ms,
            start_bin=start_bin,
            num_bins=num_bins,
            subsample=subsample,
            agg_start_x=agg_start_x,
            agg_start_y=agg_start_y,
            agg_merge_x=agg_merge_x,
            agg_merge_y=agg_merge_y,
        )
        self._write_queue.put(packed)

        self._reader = multiprocessing.Process(
            target=self._read_serial_background,
            args=(
                port,
                BAUDRATE,
                serial_wait_s,
                self._stop_event,
                self._initialized_event,
                self._queue,
                self._write_queue,
            ),
            daemon=True,
        )
        self._reader.start()
        self._initialized_event.wait()

    @staticmethod
    def _read_serial_background(
        port: str | None,
        baudrate: int,
        wait_s: float,
        stop_event: multiprocessing.synchronize.Event,
        init_event: multiprocessing.synchronize.Event,
        rx_queue: multiprocessing.Queue,
        tx_queue: multiprocessing.Queue,
    ) -> None:
        try:
            chosen = port if port is not None else find_first_port()
            conn = serial.Serial(chosen, baudrate=baudrate, timeout=1)
            conn.flush()
            time.sleep(wait_s)
        except Exception:
            stop_event.set()
            init_event.set()
            return

        # Drain anything still buffered from a previous run, then send the
        # initial config BEFORE we start surfacing frames to the parent. This
        # prevents the parent from consuming stale-config frames.
        try:
            if hasattr(conn, "reset_input_buffer"):
                conn.reset_input_buffer()
            try:
                initial_cfg = tx_queue.get_nowait()
                conn.write(initial_cfg)
            except multiprocessing.queues.Empty:
                pass
            # Give the sensor a beat to reconfigure before we start reading.
            time.sleep(wait_s)
            if hasattr(conn, "reset_input_buffer"):
                conn.reset_input_buffer()
        except Exception:
            stop_event.set()
            init_event.set()
            if conn.is_open:
                conn.close()
            return

        init_event.set()

        try:
            while not stop_event.is_set():
                line = conn.readline()
                if line:
                    try:
                        rx_queue.put_nowait(line)
                    except multiprocessing.queues.Full:
                        pass
                try:
                    payload = tx_queue.get_nowait()
                    conn.write(payload)
                except multiprocessing.queues.Empty:
                    pass
        except Exception:
            stop_event.set()
        finally:
            if conn.is_open:
                conn.close()

    def accumulate(self) -> dict[str, np.ndarray]:
        """Block until one complete frame is assembled, then return it."""
        self._data.reset()
        began = False
        while not self._data.is_done:
            try:
                raw: bytes = self._queue.get(timeout=1)
            except multiprocessing.queues.Empty:
                continue

            try:
                line = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue

            if line.startswith("D"):
                began = True
                continue

            if not began:
                continue

            tokens = [tok.strip() for tok in line.split(" ") if tok.strip()]
            if not self._data.process_row(tokens):
                self._data.reset()
                began = False

        hist = self._data.histogram()
        # The L8CH firmware emits a per-pixel distance (mm) in row[2]; use it
        # directly. _compute_distance from the histogram is a fallback only.
        dist = self._data.distance()
        pc = _compute_point_cloud(dist, fovx=self.fovx, fovy=self.fovy)
        return {"histogram": hist, "distance": dist, "point_cloud": pc}

    @property
    def is_okay(self) -> bool:
        return (
            self._initialized_event.is_set()
            and self._reader.is_alive()
            and not self._stop_event.is_set()
        )

    def close(self) -> None:
        """Signal the background reader to stop and join it.

        Safe to call multiple times.
        """
        if not hasattr(self, "_stop_event"):
            return
        self._stop_event.set()
        if not self._initialized_event.is_set():
            return
        self._reader.join(timeout=5)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
