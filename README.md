# Plug-and-Play Consumer Non-Line-of-Sight Imaging

Self-contained, plug-and-play demo of non-line-of-sight (NLOS) tracking with
a single time-of-flight SPAD. Captures per-pixel timing histograms from a
VL53L8CH sensor, runs a particle filter against a precomputed canonical
forward model, and renders a live top-down view of the hidden object's
position.

```
.
├── config.py          # single Config dataclass (all tunables)
├── spad_driver.py     # VL53L8CH driver
├── sensor.py          # build_sensor + capture helpers
├── particle_filter.py # ParticleFilter + forward model + scoring
├── dashboard.py       # PyQtGraph top-down view
├── calibrate.py       # entry point: capture point cloud
├── track.py           # entry point: live tracking
├── flash.py           # one-time: build + upload sensor firmware
├── canons/point.npy   # canonical-object point cloud (shared with paper/)
├── firmware/vl53l8ch/ # STM32 firmware source tree
└── paper/             # code + data accompanying the paper (see below)
```

> **Looking for the paper code?** The implementation, training scripts,
> configs, and captured datasets used in *Imaging Hidden Objects with
> Consumer LiDAR* live in [paper/](paper/). The rest of this README
> documents the plug-and-play demo at the repo root. See
> [paper/README.md](paper/README.md) for setup and entry points
> (`tracking.py`, `cam_localization.py`, `reconstruction.py`).


## 1. Buy the sensor

This demo targets the **P-NUCLEO-53L8A1** evaluation kit from ST:

> https://www.st.com/en/evaluation-tools/p-nucleo-53l8a1.html

The kit ships as a Nucleo-F401RE host board with the X-NUCLEO-53L8A1
expansion shield containing a histogram-capable VL53L8 multi-zone SPAD. The
firmware we flash later assumes exactly this hardware.


## 2. Hardware

- The P-NUCLEO-53L8A1 from step 1.
- USB cable connecting the Nucleo to your Mac.
- A flat wall in the sensor's field of view (calibration assumes the FOV is
  fully filled by a planar surface — see the tips section at the bottom).


## 3. ARM cross-compiler (one-time, for firmware flashing only)

You only need this if the sensor hasn't been flashed yet. If you already have
a working sensor, skip to section 4.

### Install

Download the latest macOS ARM bare-metal toolchain from
https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads.

- **Apple silicon**: pick `arm-gnu-toolchain-*-darwin-arm64-arm-none-eabi.tar.xz`
- **Intel Mac**:    pick `arm-gnu-toolchain-*-darwin-x86_64-arm-none-eabi.tar.xz`

Extract it:

```bash
mkdir -p ~/arm-toolchain
cd ~/arm-toolchain
tar -xf ~/Downloads/arm-gnu-toolchain-*-darwin-*-arm-none-eabi.tar.xz
```

Add its `bin/` to your shell PATH. Note the exact version string
(`ls ~/arm-toolchain` will show it — e.g. `arm-gnu-toolchain-15.2.rel1-...`):

```bash
echo 'export PATH="$HOME/arm-toolchain/arm-gnu-toolchain-15.2.rel1-darwin-arm64-arm-none-eabi/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Verify:

```bash
which arm-none-eabi-gcc
arm-none-eabi-gcc --version
```

If macOS Gatekeeper blocks it, strip the quarantine attribute:

```bash
xattr -dr com.apple.quarantine ~/arm-toolchain
```

### Flash the sensor

1. Plug the Nucleo in via USB. It should mount as a drive named `NOD_F401RE`
   (verify with `ls /Volumes/`). If it doesn't, hold the user button while
   pressing reset, or it auto-enters DFU mode when no firmware is loaded.

2. Run:

```bash
python flash.py
```

The script auto-detects the mounted board, runs `make clean all` in
`firmware/vl53l8ch/build/`, and copies the resulting `.bin` to the board.
First build takes ~1–2 minutes; subsequent uploads (with `--no-build`) are
near-instant.

If auto-detection fails, pass the mount path explicitly:

```bash
python flash.py /Volumes/NOD_F401RE
```


## 4. Python environment

Tested with Python 3.11+. Either a conda env or a venv works.

```bash
# conda
conda create -n nlos python=3.11
conda activate nlos

# or venv
python3.11 -m venv ~/venvs/nlos
source ~/venvs/nlos/bin/activate
```

Install the runtime dependencies:

```bash
pip install -r requirements.txt
```

The Qt binding is needed by `pyqtgraph` for the live dashboard — `PyQt6` is
the recommended choice; `PySide6` also works.


## 5. Calibrate the wall

Point the SPAD at the planar wall, plug it in, then:

```bash
python calibrate.py
```

This captures the wall point cloud and saves it to `logs/pt_cloud/`. As a
sanity check, confirm the reported distance to the wall roughly matches the
real-world distance.


## 6. Run live tracking

```bash
python track.py
```

Walkthrough:

1. Loads `logs/pt_cloud/calibration.npz`.
2. Prompts you to press Enter, then captures ~2 s of stationary frames for
   a **background histogram** (the room with no hidden object present).
   The 1-bounce peak region is masked out per pixel.
   Saves `logs/background/calibration.npz`.
3. Prompts you again to start the live capture. Step the hidden object into
   the scene now.
4. Voxelizes the canonical forward model (takes ~30 s on CPU on first run).
5. Opens the fullscreen dashboard. Each frame: background-subtract, mask 1B,
   shift the per-pixel 1B peak to bin 0, run a particle-filter update,
   redraw.

Dashboard controls (buttons, top-left):

- **show/hide particles** — toggle the full cloud (mean dot + trail always
  shown).
- **show/hide grid lines** — toggle plot grid.
- **flip x/z axes** — useful if your sensor mounts vertically.

Press Ctrl-C in the terminal or close the window to stop. The sensor's
serial port is released cleanly on every exit path.


## 7. Tuning

All knobs live in [config.py](config.py). Common ones:

| Field | Effect |
|---|---|
| `num_bins`, `start_bin_*`, `ranging_frequency_hz` | Sensor capture window. |
| `background_seconds` | Number of frames averaged for calibration / background. |
| `pulse_half_width` | How aggressively the 1B peak is masked. |
| `num_particles`, `radius`, `eta` | Particle-filter aggressiveness. |
| `x_range`, `y_range`, `z_range` | Hidden-object search volume (m). |
| `canon_*` | Forward-model grid extent and resolution. |
| `dashboard_xlim`, `occluder_x` | Dashboard layout. |
| `fullscreen` | Set to `False` if you want a windowed dashboard. |

After changing config, re-run from `calibrate.py` if you touched sensor
fields, otherwise `track.py` alone is enough.


## Troubleshooting

**`/bin/sh: arm-none-eabi-gcc: command not found`** during `python flash.py`
— PATH isn't pointing at the toolchain. `which arm-none-eabi-gcc` should
return the binary in your `~/arm-toolchain/...-rel1-.../bin/` directory.
Source `~/.zshrc` again or check the version string in the export line.

**`python flash.py` runs but prints nothing for a while** — the build is
working, just verbose. Compiles ~80 source files on first run; expect
~1–2 minutes. Each line shows the file being built. To go silent, pass
`-q`.

**`cannot reshape array of size N into shape (4,4,B)`** during calibration
— the sensor is streaming with a stale firmware config. This was fixed in
the driver (it now drains the buffer + waits after sending config), but if
you still see it, the row-length guard in the accumulator will silently
skip mismatched frames. Just wait a second and try again; the sensor
should converge to the requested config within a few seconds.

**Distances are exactly 1/2 of reality** — the host-side
`_compute_distance` fallback got engaged instead of the firmware-emitted
distance. Make sure the firmware was flashed successfully and the sensor
is sending the standard line format (idx ambient distance bin0 bin1 …).

**Dashboard crashes with a Qt error** — `pip install PyQt6` (or
`PySide6`). pyqtgraph alone is not enough.

**Serial port held open after a crash** — the driver's `close()` joins the
multiprocessing reader, which releases the port. If a hard kill leaves it
held, unplug and replug the USB cable.


## Tips for setting up your hardware

A few practical notes from running this demo. The geometry of your scene
matters at least as much as the code; if results look bad, suspect the setup
first.

**Start with a known-good geometry, then adapt.**
The easiest way to get a working setup on day one is to copy a canonical
around-the-corner arrangement: the SPAD pointed at a flat relay wall, an
occluder between the sensor and the hidden region (so the sensor cannot see
the hidden object directly), and the hidden object somewhere in the
occluded volume behind the occluder. Once that produces sensible tracking,
modify one variable at a time toward whatever scene you actually care
about. Trying to debug both code and an unusual scene at once is hard.

**Trade angle for resolution vs. SNR.**
There's a real tradeoff in how you aim the SPAD at the relay wall:

- Aim *more obliquely* → the FOV covers a larger patch of wall → more
  spatial diversity in the per-pixel histograms → better reconstruction
  resolution, *but* the photon return per pixel drops and SNR suffers.
- Aim *closer to perpendicular* and at a nearer patch → smaller wall area
  per pixel → tighter, brighter return → much better SNR, *but* less
  spatial information for the particle filter to lock onto.

Play with the angle and the sensor-to-wall distance until you find a sweet
spot. Watch the calibration histogram plot: you want a clear, well-defined
1-bounce stripe on every pixel, not a noisy floor.

**Stay inside the SPAD's range.**
Total round-trip range for the VL53L8CH at the default `num_bins=48`,
`subsample=1`, 250 ps/bin is around 4 m. To leave headroom for both the
1-bounce return *and* the much weaker 3-bounce return from the hidden
object, a good rule of thumb is:

- **SPAD ↔ relay wall**: < 1 m.
- **Relay wall ↔ hidden object**: ~1–1.5 m.

If the hidden object is too far, its 3-bounce return falls outside the
timing gate and the particle filter has nothing to score against. Too
close, and its return overlaps the 1-bounce peak and gets masked out
during preprocessing.

**Lighting and reflectivity.**
A matte, light-colored relay wall is ideal. Glossy or dark walls deflect
or absorb too many photons. Background ambient light isn't a major issue
indoors at typical office levels — the histogram subtraction step in
`track.py` handles it — but direct sunlight on the wall will saturate the
sensor.


## Citation

If you use this code, we'd appreciate it if you could cite our paper:

```bibtex
@article{somasundaram2026imaging,
  title   = {Imaging hidden objects with consumer LiDAR via motion induced sampling},
  author  = {Somasundaram, Siddharth and Young, Aaron and Dave, Akshat and Pediredla, Adithya and Raskar, Ramesh},
  journal = {Nature},
  year    = {2026},
  doi     = {10.1038/s41586-026-10502-x},
  url     = {https://www.nature.com/articles/s41586-026-10502-x}
}
```
