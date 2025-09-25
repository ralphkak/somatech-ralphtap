# brightness_roi_plotter.py (adds live FFT plot of selected region)
# Requirements: PyQt6, pyqtgraph, opencv-python, numpy

import sys
import os
import math
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QPushButton, QHBoxLayout,
    QVBoxLayout, QLabel, QSplitter, QProgressBar, QMessageBox, QCheckBox
)
import pyqtgraph as pg


def rgb_to_luma(frame_bgr: np.ndarray) -> np.ndarray:
    b = frame_bgr[..., 0].astype(np.float32)
    g = frame_bgr[..., 1].astype(np.float32)
    r = frame_bgr[..., 2].astype(np.float32)
    return 0.0722 * b + 0.7152 * g + 0.2126 * r


class VideoWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray, np.ndarray, float)  # times, values, fps
    error = pyqtSignal(str)

    def __init__(self, path: str, roi_rect: QRectF | None, use_roi: bool):
        super().__init__()
        self.path = path
        self.roi_rect = roi_rect
        self.use_roi = use_roi
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                self.error.emit("Could not open video.")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or math.isnan(fps):
                fps = 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                total = None

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self.use_roi and self.roi_rect is not None:
                x0 = max(0, int(self.roi_rect.left()))
                y0 = max(0, int(self.roi_rect.top()))
                x1 = min(width, int(self.roi_rect.right()))
                y1 = min(height, int(self.roi_rect.bottom()))
                if x1 <= x0 or y1 <= y0:
                    self.error.emit("ROI is empty; please select a non-zero area.")
                    cap.release()
                    return
                roi_slice = (slice(y0, y1), slice(x0, x1))
            else:
                roi_slice = (slice(0, height), slice(0, width))

            ts, vals = [], []
            frame_idx = 0
            last_emit = 0

            while True:
                if self._stop:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                t = (t_ms / 1000.0) if (t_ms and t_ms > 0) else (frame_idx / fps)

                y = rgb_to_luma(frame)
                vals.append(float(y[roi_slice].mean()))
                ts.append(t)

                frame_idx += 1
                if total:
                    prog = int(100 * frame_idx / total)
                    if prog - last_emit >= 1:
                        last_emit = prog
                        self.progress.emit(prog)

            cap.release()

            if len(ts) == 0:
                self.error.emit("No frames read from video.")
                return

            t_arr = np.array(ts, dtype=np.float64)
            v_arr = np.array(vals, dtype=np.float32)

            _, unique_idx = np.unique(t_arr, return_index=True)
            unique_idx = np.sort(unique_idx)
            t_arr = t_arr[unique_idx]
            v_arr = v_arr[unique_idx]

            self.progress.emit(100)
            self.finished.emit(t_arr, v_arr, fps)

        except Exception as e:
            self.error.emit(str(e))


class BrightnessROIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MP4 Brightness ROI Plotter (PyQt6 + pyqtgraph)")
        self.resize(1300, 800)

        # State
        self.video_path: str | None = None
        self.first_frame: np.ndarray | None = None
        self.times: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.fps: float | None = None
        self.worker: VideoWorker | None = None
        self.selection = None
        self.start_end_defined = False
        self.roi_added = False

        # === Top bar ===
        top_bar = QHBoxLayout()
        self.btn_load = QPushButton("Load MP4")
        self.btn_analyze = QPushButton("Analyze (ROI → Brightness vs Time)")
        self.btn_analyze.setEnabled(False)
        self.btn_clear_sel = QPushButton("Clear Selection")
        self.btn_clear_sel.setEnabled(False)
        self.chk_fullframe = QCheckBox("Use Full Frame (ignore ROI)")
        self.chk_fullframe.setChecked(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        top_bar.addWidget(self.btn_load)
        top_bar.addWidget(self.btn_analyze)
        top_bar.addWidget(self.btn_clear_sel)
        top_bar.addWidget(self.chk_fullframe)
        top_bar.addStretch()
        top_bar.addWidget(self.progress)

        # === Left panel: image + ROI ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.img_view = pg.GraphicsLayoutWidget()
        self.img_plot = self.img_view.addPlot()
        self.img_plot.invertY(True)
        self.img_plot.setAspectLocked(True)
        self.img_item = pg.ImageItem()
        self.img_plot.addItem(self.img_item)
        self.roi = pg.RectROI([20, 20], [100, 100], pen=pg.mkPen((0, 255, 0), width=2))
        self.roi.setZValue(10)
        self.roi.sigRegionChanged.connect(self._on_roi_changed)
        self.status = QLabel("Load a video to begin.")
        self.status.setWordWrap(True)
        left_layout.addWidget(self.img_view)
        left_layout.addWidget(self.status)

        # === Right panel: time plot + FFT plot ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Time series plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Avg Brightness', units='')
        self.curve = self.plot_widget.plot([], [], pen=pg.mkPen(width=2))
        right_layout.addWidget(self.plot_widget)

        self.hover_label = QLabel("Hover: —")
        right_layout.addWidget(self.hover_label)

        self.result_label = QLabel("Selection: —")
        self.result_label.setWordWrap(True)
        right_layout.addWidget(self.result_label)

        # FFT plot (third graph)
        self.fft_plot = pg.PlotWidget()
        self.fft_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.fft_plot.setLabel('left', 'Magnitude', units='')
        self.fft_curve = self.fft_plot.plot([], [], pen=pg.mkPen(width=2))
        self.fft_note = QLabel("")  # shows messages like "No FFT"
        right_layout.addWidget(self.fft_plot)
        right_layout.addWidget(self.fft_note)

        splitter = QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addLayout(top_bar)
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

        # Crosshair on time plot
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((150, 150, 255), width=1, style=Qt.PenStyle.DashLine))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((150, 150, 255), width=1, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.vline, ignoreBounds=True)
        self.plot_widget.addItem(self.hline, ignoreBounds=True)
        self.proxy_move = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_plot_clicked)

        # Signals
        self.btn_load.clicked.connect(self.load_video)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_clear_sel.clicked.connect(self.clear_selection)
        self.chk_fullframe.toggled.connect(self._on_fullframe_toggle)

    # ---------------------------- UI Actions ----------------------------

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open MP4", "", "Video Files (*.mp4 *.mov *.m4v *.avi);;All Files (*)")
        if not path:
            return
        self.video_path = path

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video.")
            return
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            QMessageBox.critical(self, "Error", "Could not read first frame.")
            return

        self.first_frame = frame
        self._show_frame(frame)
        self._add_roi_once()
        self.status.setText(f"Loaded: {os.path.basename(path)} — draw/resize the green ROI or tick 'Use Full Frame', then click Analyze.")
        self.btn_analyze.setEnabled(True)
        self.progress.setValue(0)
        self.times = None
        self.values = None
        self.curve.setData([], [])
        self._clear_fft_plot()

    def _show_frame(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.img_item.setImage(rgb, levels=(0, 255))
        h, w, _ = rgb.shape
        self.img_item.setRect(QRectF(0, 0, w, h))
        self.img_plot.setXRange(0, w)
        self.img_plot.setYRange(0, h)

    def _add_roi_once(self):
        if not self.roi_added:
            self.img_plot.addItem(self.roi)
            self.roi_added = True

    def _on_roi_changed(self):
        if self.first_frame is None:
            return
        h, w, _ = self.first_frame.shape
        rect = self.roi.parentBounds().intersected(QRectF(0, 0, w, h))
        self.roi.setPos([rect.left(), rect.top()])
        self.roi.setSize([max(1, rect.width()), max(1, rect.height())])

    def _on_fullframe_toggle(self, checked: bool):
        self.roi.setVisible(not checked)

    def run_analysis(self):
        if not self.video_path:
            QMessageBox.information(self, "Info", "Please load a video first.")
            return
        use_roi = not self.chk_fullframe.isChecked()
        roi_rect = self.roi.parentBounds() if (use_roi and self.roi is not None) else None

        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        self.progress.setValue(0)
        self.status.setText("Analyzing…")
        self.worker = VideoWorker(self.video_path, roi_rect, use_roi)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.error.connect(self._on_worker_error)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.start()

    def _on_worker_error(self, msg: str):
        self.status.setText("Error.")
        QMessageBox.critical(self, "Error", msg)

    def _on_worker_done(self, times: np.ndarray, values: np.ndarray, fps: float):
        self.times = times
        self.values = values
        self.fps = fps
        self.curve.setData(self.times, self.values)
        self.plot_widget.enableAutoRange()
        self.status.setText(f"Done. Frames: {len(times)} | FPS≈{fps:.3f}. Hover to inspect; click to set start/end and compute frequency.")

        self._ensure_selection_item()
        t0 = float(self.times[0])
        # default: first 20% of duration (or first two samples)
        t1 = float(self.times[min(len(self.times) - 1, max(1, len(self.times)//5))])
        if t1 <= t0 and len(self.times) > 1:
            t1 = float(self.times[1])
        self.selection.setRegion((t0, t1))
        self.btn_clear_sel.setEnabled(True)
        self.result_label.setText("Selection: drag the shaded region or click twice on the plot to set exact start/end.")
        # Update FFT for the default region
        self._compute_selection_stats((t0, t1))

    def _ensure_selection_item(self):
        if self.selection is None:
            self.selection = pg.LinearRegionItem()
            self.selection.setZValue(10)
            self.selection.sigRegionChanged.connect(self._on_region_changed)
            self.plot_widget.addItem(self.selection)

    def clear_selection(self):
        if self.selection:
            self.plot_widget.removeItem(self.selection)
            self.selection = None
        self.result_label.setText("Selection: —")
        self.btn_clear_sel.setEnabled(False)
        self.start_end_defined = False
        self._clear_fft_plot()

    # ---------------------------- Plot interactivity ----------------------------

    def _on_mouse_moved(self, evt):
        if self.times is None or self.values is None:
            return
        pos = evt[0]
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            return
        vb = self.plot_widget.getPlotItem().vb
        mouse_point = vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()
        self.vline.setPos(x)
        self.hline.setPos(y)
        idx = np.searchsorted(self.times, x)
        idx = 0 if idx <= 0 else (len(self.times) - 1 if idx >= len(self.times) else idx)
        t = self.times[idx]
        v = self.values[idx]
        self.hover_label.setText(f"Hover: t={t:.3f} s, value={v:.3f}")

    def _on_plot_clicked(self, mouse_event):
        if mouse_event.button() != Qt.MouseButton.LeftButton:
            return
        vb = self.plot_widget.getPlotItem().vb
        if not vb.geometry().contains(mouse_event.scenePos().toPoint()):
            return
        if self.times is None or self.values is None:
            return

        mouse_point = vb.mapSceneToView(mouse_event.scenePos())
        x = float(mouse_point.x())

        self._ensure_selection_item()
        region = list(self.selection.getRegion())
        if not self.start_end_defined:
            region[0] = x
            region[1] = x
            self.selection.setRegion(tuple(sorted(region)))
            self.start_end_defined = True
        else:
            region[1] = x
            region = sorted(region)
            self.selection.setRegion(tuple(region))
            self.start_end_defined = False
            self._compute_selection_stats(tuple(region))

    def _on_region_changed(self):
        if self.selection is None or self.times is None:
            return
        region = tuple(sorted(self.selection.getRegion()))
        self._compute_selection_stats(region)

    # ---------------------------- Computation & FFT plot ----------------------------

    def _clear_fft_plot(self):
        self.fft_curve.setData([], [])
        self.fft_note.setText("")

    def _compute_selection_stats(self, region: tuple[float, float]):
        if self.times is None or self.values is None:
            return
        t0, t1 = region
        if t1 <= t0:
            self.result_label.setText("Selection: invalid (end ≤ start).")
            self._clear_fft_plot()
            return

        left = np.searchsorted(self.times, t0, side='left')
        right = np.searchsorted(self.times, t1, side='right')
        seg_t = self.times[left:right]
        seg_v = self.values[left:right]

        duration = float(seg_t[-1] - seg_t[0]) if len(seg_t) > 1 else 0.0
        if len(seg_t) < 4 or duration <= 0:
            self.result_label.setText(f"Selection: {t1 - t0:.3f}s (too few samples for FFT).")
            self._clear_fft_plot()
            self.fft_note.setText("No FFT (too few samples)")
            return

        dt = np.diff(seg_t)
        median_dt = float(np.median(dt)) if dt.size else (1.0 / (self.fps if self.fps else 30.0))
        fs = 1.0 / median_dt if median_dt > 0 else (self.fps if self.fps else 30.0)

        # Detrend and window to reduce spectral leakage
        x = seg_v - np.mean(seg_v)
        n = len(x)
        window = np.hanning(n)
        xw = x * window

        freq = np.fft.rfftfreq(n, d=1.0 / fs)
        spec = np.fft.rfft(xw)
        mag = np.abs(spec)

        # Exclude DC when finding peak
        if len(freq) > 1:
            peak_idx = int(np.argmax(mag[1:]) + 1)
            dom_f = float(freq[peak_idx])
            dom_mag = float(mag[peak_idx])
        else:
            dom_f = 0.0
            dom_mag = 0.0

        self.result_label.setText(
            f"Selection: {duration:.3f} s  |  Fs≈{fs:.2f} Hz  |  Dominant freq≈{dom_f:.3f} Hz (mag {dom_mag:.3f})"
        )

        # Update FFT graph (skip DC bin to emphasize oscillations)
        if len(freq) >= 2:
            self.fft_curve.setData(freq[1:], mag[1:])
            self.fft_plot.enableAutoRange()
            self.fft_note.setText("")
        else:
            self._clear_fft_plot()
            self.fft_note.setText("No FFT")

def main():
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    win = BrightnessROIApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
