import matplotlib.cm as cm
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from config import Config


class _CameraWidget(QtWidgets.QWidget):
    """Small overlay widget that renders a 2D image (used for the raw signal)."""

    def __init__(self, parent=None, size=(320, 240)):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setFixedSize(*size)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setScaledContents(True)
        layout.addWidget(self.label)

    def update_image(self, image: np.ndarray) -> None:
        if image.ndim == 2:
            norm = (image - image.min()) / (image.max() - image.min() + 1e-12)
            image = (cm.hot(norm)[:, :, :3] * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.interp(
                image, (image.min(), image.max()), (0, 255)
            ).astype(np.uint8)
        h, w, _ = image.shape
        qimg = QtGui.QImage(
            image.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888
        )
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setBrush(QtGui.QColor(255, 255, 255, 100))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect(), 6, 6)


class _ToggleButton(QtWidgets.QWidget):
    """Generic show/hide button — calls a callback with the new on/off state."""

    def __init__(self, parent, label: str, on_toggle, initial: bool = True):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.button = QtWidgets.QPushButton(label)
        layout.addWidget(self.button)
        self._state = initial
        self._on_toggle = on_toggle
        self.button.clicked.connect(self._clicked)

    def _clicked(self):
        self._state = not self._state
        self._on_toggle(self._state)


class Dashboard:
    """Top-down PyQtGraph view: relay wall, sensor pose/FOV, occluder,
    particles + mean + trail, raw-signal overlay."""

    def __init__(self, cfg: Config, cam_z: float, z_range: tuple[float, float]):
        self.cfg = cfg
        self.cam_z = cam_z
        self.flip_xz = False
        self._last_particles: np.ndarray | None = None
        self._past_x: list[float] = []
        self._past_z: list[float] = []
        self._box_sz = 0.1
        self._triangle_h = 0.05
        self._wall_thickness = 0.1

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.mkQApp()

        self.win = pg.GraphicsLayoutWidget(title="Volume Projection", show=True)
        self.plot = self.win.addPlot()
        self.plot.showGrid(x=True, y=True)
        self.plot.setAspectLocked()
        self.plot.setLabels(bottom="X (m)", left="Z (m)")
        self.plot.setXRange(cfg.dashboard_xlim[0], cfg.dashboard_xlim[1])
        self.plot.setYRange(z_range[0] - self._wall_thickness, z_range[1])
        self.plot.invertY(True)

        grey = pg.mkPen((200, 200, 200), width=1, style=QtCore.Qt.PenStyle.DashLine)
        self.plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=grey))
        self.plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=grey))

        self._add_relay_wall()
        self._add_sensor()
        self._add_occluder()
        self._add_particles()
        self.sig_widget = _CameraWidget(self.win)
        self._add_toggles()

        if cfg.fullscreen:
            self.win.showFullScreen()
        self.sig_widget.show()
        self._place_sig_widget()

        orig_resize = self.win.resizeEvent

        def _on_resize(ev):
            orig_resize(ev)
            self._place_sig_widget()

        self.win.resizeEvent = _on_resize

    # ---- layout ----

    def _add_relay_wall(self) -> None:
        xlo, xhi = self.cfg.dashboard_xlim
        rect = QtGui.QPainterPath()
        rect.addRect(xlo, -self._wall_thickness, xhi - xlo, self._wall_thickness)
        item = QtWidgets.QGraphicsPathItem(rect)
        item.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
        item.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self.plot.addItem(item)

        label = pg.TextItem("Relay Wall", anchor=(0.5, 0.5), color="k")
        font = QtGui.QFont()
        font.setPointSize(30)
        font.setBold(True)
        label.setFont(font)
        label.setPos((xlo + xhi) / 2, -self._wall_thickness - 0.05)
        self.plot.addItem(label)

    def _add_sensor(self) -> None:
        self.sensor_box = QtWidgets.QGraphicsRectItem(
            -self._box_sz / 2, -self._box_sz / 2, self._box_sz, self._box_sz
        )
        self.sensor_box.setBrush(QtGui.QBrush(QtGui.QColor(128, 128, 128)))
        self.sensor_box.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self.plot.addItem(self.sensor_box)

        self.sensor_triangle = QtWidgets.QGraphicsPolygonItem(
            QtGui.QPolygonF([
                QtCore.QPointF(0, self._triangle_h),
                QtCore.QPointF(-self._triangle_h, 0),
                QtCore.QPointF(self._triangle_h, 0),
            ])
        )
        self.sensor_triangle.setBrush(QtGui.QBrush(QtGui.QColor(128, 128, 128)))
        self.sensor_triangle.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self.sensor_triangle.setTransformOriginPoint(
            0, self._triangle_h + self._box_sz / 2
        )
        self.plot.addItem(self.sensor_triangle)

        font = QtGui.QFont()
        font.setPointSize(30)
        font.setBold(True)
        self.sensor_label = pg.TextItem("Sensor", anchor=(0.5, 0.0), color="gray")
        self.sensor_label.setFont(font)
        self.plot.addItem(self.sensor_label)

        pen_fov = pg.mkPen(
            (128, 128, 128), width=3, style=QtCore.Qt.PenStyle.DashLine
        )
        self.fov_left = QtWidgets.QGraphicsLineItem()
        self.fov_right = QtWidgets.QGraphicsLineItem()
        self.fov_left.setPen(pen_fov)
        self.fov_right.setPen(pen_fov)
        self.plot.addItem(self.fov_left)
        self.plot.addItem(self.fov_right)

    def _add_occluder(self) -> None:
        line = QtWidgets.QGraphicsLineItem()
        line.setPen(pg.mkPen((0, 0, 0), width=3, style=QtCore.Qt.PenStyle.SolidLine))
        line.setLine(
            self.cfg.occluder_x,
            self.cam_z,
            self.cfg.occluder_x,
            self.cam_z + self.cfg.occluder_wall_length,
        )
        self.plot.addItem(line)

        font = QtGui.QFont()
        font.setPointSize(30)
        font.setBold(True)
        label = pg.TextItem("Occluder", anchor=(0.5, 0.0), color="black")
        label.setFont(font)
        label.setPos(
            -(-self.cfg.occluder_x + 0.2) - 0.05,
            self.cam_z + self.cfg.occluder_wall_length * 3 / 4,
        )
        self.plot.addItem(label)

    def _add_particles(self) -> None:
        self.particles = pg.ScatterPlotItem()
        self.plot.addItem(self.particles)

        self.mean_particle = pg.ScatterPlotItem(
            size=30, pen=None, brush=pg.mkBrush(255, 0, 0, 255)
        )
        self.plot.addItem(self.mean_particle)

        self.past_mean = pg.PlotCurveItem(
            pen=pg.mkPen(color=(255, 0, 0, 255), width=20)
        )
        self.plot.addItem(self.past_mean)

    def _add_toggles(self) -> None:
        self.toggle_particles = _ToggleButton(
            self.win,
            "show/hide particles",
            lambda on: self.particles.setVisible(on),
            initial=False,
        )
        self.toggle_particles.show()

        self.toggle_grid = _ToggleButton(
            self.win,
            "show/hide grid lines",
            lambda on: self.plot.showGrid(x=on, y=on),
            initial=True,
        )
        self.toggle_grid.show()
        self.toggle_grid.move(0, self.toggle_grid.height() - 15)

        self.toggle_axis = _ToggleButton(
            self.win, "flip x/z axes", self._on_flip_axes, initial=False
        )
        self.toggle_axis.show()
        self.toggle_axis.move(
            0, self.toggle_grid.height() + self.toggle_axis.height() - 15
        )

    # ---- runtime ----

    def _place_sig_widget(self) -> None:
        margin = 50
        h = self.win.size().height()
        w = self.win.size().width()
        self.sig_widget.move(
            w - self.sig_widget.width() - margin,
            h - self.sig_widget.height() - margin,
        )

    def _map_xz(self, x, z):
        return (z, x) if self.flip_xz else (x, z)

    def _on_flip_axes(self, _on: bool) -> None:
        self.flip_xz = not self.flip_xz
        if self._last_particles is not None:
            self._draw_particles(self._last_particles)

    def _draw_particles(self, particles: np.ndarray) -> None:
        px, pz = self._map_xz(particles[:, 0], particles[:, 2])
        self.particles.setData(px, pz)

        mean = particles.mean(axis=0)
        mx, mz = self._map_xz(mean[0], mean[2])
        self.mean_particle.setData([mx], [mz])

        past_x, past_z = self._map_xz(
            np.asarray(self._past_x), np.asarray(self._past_z)
        )
        self.past_mean.setData(past_x, past_z)

    def _update_sensor_pose(self, pt_cloud: np.ndarray) -> None:
        x_min = float(np.min(pt_cloud[:, 0]))
        x_max = float(np.max(pt_cloud[:, 0]))

        self.fov_left.setVisible(True)
        self.fov_right.setVisible(True)
        self.fov_left.setLine(x_min, 0, 0, self.cam_z)
        self.fov_right.setLine(x_max, 0, 0, self.cam_z)

        ray_1 = np.array([x_min, 0]) - np.array([0, self.cam_z])
        ray_2 = np.array([x_max, 0]) - np.array([0, self.cam_z])
        ray_1 /= np.linalg.norm(ray_1)
        ray_2 /= np.linalg.norm(ray_2)
        cam_ray = (ray_1 + ray_2) / 2
        rot = np.degrees(np.arctan2(cam_ray[1], cam_ray[0])) + 90

        self.sensor_box.setRotation(rot)
        self.sensor_triangle.setRotation(rot)

        box_pos = np.array([0, self.cam_z]) - cam_ray * (
            self._box_sz / 2 + self._triangle_h
        )
        tri_pos = np.array(
            [0, self.cam_z - self._triangle_h - self._box_sz / 2]
        ) - cam_ray * (self._box_sz / 2 + self._triangle_h)
        self.sensor_box.setPos(box_pos[0], box_pos[1])
        self.sensor_triangle.setPos(tri_pos[0], tri_pos[1])
        self.sensor_label.setPos(-(self._box_sz + 0.15) - 0.05, self.cam_z)

    def update(
        self,
        particles: np.ndarray,
        signal: np.ndarray,
        pt_cloud: np.ndarray,
    ) -> None:
        self._last_particles = particles

        mean = particles.mean(axis=0)
        self._past_x.append(float(mean[0]))
        self._past_z.append(float(mean[2]))
        if len(self._past_x) >= self.cfg.trail_length:
            self._past_x.pop(0)
            self._past_z.pop(0)

        self._draw_particles(particles)
        self.sig_widget.update_image(signal)
        self._update_sensor_pose(pt_cloud)
        pg.QtGui.QGuiApplication.processEvents()

    @property
    def is_okay(self) -> bool:
        return self.win.isVisible()

    def close(self) -> None:
        self.win.close()
