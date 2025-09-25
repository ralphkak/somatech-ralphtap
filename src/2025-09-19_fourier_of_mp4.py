import sys
import os
import csv
import cv2
import numpy as np

from PyQt6.QtCore import Qt, QThread, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QAction, QPixmap, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QGridLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QMessageBox, QGroupBox, QStatusBar
)

# ---- Qt Charts (native PyQt6 plotting) ----
from PyQt6.QtCharts import (
    QChart, QChartView, QLineSeries, QValueAxis
)


def cvimg_to_qpixmap(img_bgr):
    if img_bgr is None:
        return QPixmap()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class VideoFFTWorker(QThread):
    progress = pyqtSignal(str)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, path, frame_step=1, blur_sigma=0.0, roi_rect=None, use_diff=False,
                 win_sec=2.0, hop_sec=0.25):
        super().__init__()
        self.path = path
        self.frame_step = max(1, int(frame_step))
        self.blur_sigma = float(blur_sigma)
        self.roi_rect = roi_rect
        self.use_diff = bool(use_diff)
        self.win_sec = float(win_sec)
        self.hop_sec = float(hop_sec)
        self._stop = False

    def stop(self):  # not used, but handy
        self._stop = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                self.error.emit("Failed to open video.")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
                self.progress.emit("Warning: FPS reported as 0; using 30.0 as fallback.")

            # Gaussian kernel size
            ksize = 0
            if self.blur_sigma > 0:
                ksize = int(np.ceil(self.blur_sigma * 6)) | 1  # odd

            values = []
            idx = 0
            prev_gray = None
            self.progress.emit("Reading frames…")

            while True:
                if self._stop:
                    return
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if idx % self.frame_step == 0:
                    if self.roi_rect is not None and not self.roi_rect.isNull():
                        x = max(0, self.roi_rect.x())
                        y = max(0, self.roi_rect.y())
                        w = max(1, self.roi_rect.width())
                        h = max(1, self.roi_rect.height())
                        H, W = frame.shape[:2]
                        frame = frame[y:min(H, y + h), x:min(W, x + w)]

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if ksize >= 3:
                        gray = cv2.GaussianBlur(gray, (ksize, ksize), self.blur_sigma)

                    if self.use_diff:
                        if prev_gray is not None:
                            diff = cv2.absdiff(gray, prev_gray)
                            values.append(float(diff.mean()))
                        prev_gray = gray
                    else:
                        values.append(float(gray.mean()))
                idx += 1

            cap.release()

            if len(values) < 8:
                self.error.emit("Not enough samples to compute FFT (need ≥ 8). Try frame step = 1 or a longer clip.")
                return

            values = np.asarray(values, dtype=np.float64)
            eff_fps = fps / self.frame_step
            dur_s = values.size / eff_fps

            # --- Global spectrum
            t = np.arange(values.size)
            p = np.polyfit(t, values, 1)
            vals_d = values - np.polyval(p, t)
            win_g = np.hanning(values.size)
            xw = vals_d * win_g
            X = np.fft.rfft(xw)
            freqs = np.fft.rfftfreq(values.size, d=1.0 / eff_fps)
            amp = (2.0 / win_g.sum()) * np.abs(X)

            # --- Dominant frequency vs time (sliding)
            win_n = max(8, int(round(self.win_sec * eff_fps)))
            hop_n = max(1, int(round(self.hop_sec * eff_fps)))
            if win_n > values.size:
                win_n = max(8, values.size // 2)

            hann = np.hanning(win_n)
            dom_times, dom_freqs = [], []

            start = 0
            while start + win_n <= values.size:
                seg = values[start:start + win_n]
                tseg = np.arange(seg.size)
                pseg = np.polyfit(tseg, seg, 1)
                segd = seg - np.polyval(pseg, tseg)
                xw = segd * hann
                Xw = np.fft.rfft(xw)
                f = np.fft.rfftfreq(seg.size, d=1.0 / eff_fps)
                A = (2.0 / hann.sum()) * np.abs(Xw)

                # drop DC
                if f.size > 1:
                    f1 = f[1:]
                    A1 = A[1:]
                else:
                    f1 = f
                    A1 = A

                if A1.size == 0 or not np.any(np.isfinite(A1)):
                    dom_f = np.nan
                else:
                    dom_f = float(f1[int(np.nanargmax(A1))])

                center_idx = start + win_n // 2
                dom_times.append(center_idx / eff_fps)
                dom_freqs.append(dom_f)
                start += hop_n

            res = {
                "freqs": freqs,
                "amplitude": amp,
                "fps": float(fps),
                "effective_fps": float(eff_fps),
                "n_samples": int(values.size),
                "duration_s": float(dur_s),
                "dom_times": np.asarray(dom_times, dtype=np.float64),
                "dom_freqs": np.asarray(dom_freqs, dtype=np.float64),
            }
            self.result.emit(res)

        except Exception as e:
            self.error.emit(f"FFT failed: {e}")


class RoiLabel(QLabel):
    roiChanged = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pixmap = None
        self._display_pixmap = None
        self._dragging = False
        self._start = QPoint()
        self._current = QPoint()
        self._roi_display = QRect()
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._img_size = None

    def setPixmap(self, pm: QPixmap) -> None:
        self._pixmap = pm
        self._update_display()
        super().setPixmap(self._display_pixmap if self._display_pixmap else pm)

    def resizeEvent(self, event):
        self._update_display()
        super().resizeEvent(event)

    def _update_display(self):
        if self._pixmap is None or self.width() <= 2 or self.height() <= 2:
            self._display_pixmap = None
            return
        pm = self._pixmap
        self._img_size = (pm.width(), pm.height())
        scaled = pm.scaled(self.width(), self.height(),
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        self._display_pixmap = scaled
        self._scale_x = (self._img_size[0] / scaled.width()) if scaled.width() > 0 else 1.0
        self._scale_y = (self._img_size[1] / scaled.height()) if scaled.height() > 0 else 1.0
        super().setPixmap(self._display_pixmap)

    def mousePressEvent(self, event):
        if self._display_pixmap is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self._dragging = True
        self._start = event.position().toPoint()
        self._current = self._start
        self.update()

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        self._current = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            rect_disp = QRect(self._start, self._current).normalized()
            x = int(rect_disp.x() * self._scale_x)
            y = int(rect_disp.y() * self._scale_y)
            w = int(rect_disp.width() * self._scale_x)
            h = int(rect_disp.height() * self._scale_y)
            rect_img = QRect(x, y, w, h)
            self._roi_display = rect_disp
            self.roiChanged.emit(rect_img)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._dragging or not self._roi_display.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            rect = QRect(self._start, self._current).normalized() if self._dragging else self._roi_display
            painter.drawRect(rect)


# --------- Qt Charts helpers ----------
def make_chart(title: str, x_label: str, y_label: str) -> QChart:
    chart = QChart()
    chart.setTitle(title)
    chart.legend().setVisible(False)

    # Axes
    ax_x = QValueAxis()
    ax_x.setTitleText(x_label)
    ax_x.setLabelFormat("%.3f")
    ax_x.setMinorTickCount(2)

    ax_y = QValueAxis()
    ax_y.setTitleText(y_label)
    ax_y.setLabelFormat("%.3f")
    ax_y.setMinorTickCount(2)

    chart.addAxis(ax_x, Qt.AlignmentFlag.AlignBottom)
    chart.addAxis(ax_y, Qt.AlignmentFlag.AlignLeft)
    return chart


def set_xy_limits(chart: QChart, x_min, x_max, y_min, y_max):
    # chart.axisX/Y() not guaranteed; fetch the first QValueAxis of each orientation
    ax_x = next((ax for ax in chart.axes(Qt.AlignmentFlag.AlignBottom)), None)
    ax_y = next((ax for ax in chart.axes(Qt.AlignmentFlag.AlignLeft)), None)
    if ax_x:
        ax_x.setRange(float(x_min), float(x_max))
    if ax_y:
        ax_y.setRange(float(y_min), float(y_max))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Temporal FFT Analyzer (PyQt6 + QtCharts)")
        self.resize(1250, 820)

        self.video_path = None
        self.preview_frame = None
        self.current_roi = None
        self.worker = None
        self.last_result = None

        # --- Left (preview + controls)
        self.preview = RoiLabel()
        self.preview.setText("Open a video to preview\n(Drag to select ROI, or leave empty for whole frame)")
        self.preview.roiChanged.connect(self.on_roi_changed)

        controls_box = QGroupBox("Analysis Controls")
        open_btn = QPushButton("Open Video…")
        open_btn.clicked.connect(self.open_video)

        analyze_btn = QPushButton("Run FFT")
        analyze_btn.clicked.connect(self.run_fft)

        export_btn = QPushButton("Export Spectrum CSV")
        export_btn.clicked.connect(self.export_csv)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 1000)
        self.step_spin.setValue(1)
        self.step_spin.setSuffix(" frame(s)")

        self.blur_spin = QDoubleSpinBox()
        self.blur_spin.setRange(0.0, 20.0)
        self.blur_spin.setSingleStep(0.5)
        self.blur_spin.setValue(0.0)
        self.blur_spin.setSuffix(" σ")

        self.use_roi_chk = QCheckBox("Use ROI (drag on preview)")
        self.use_roi_chk.setChecked(False)

        self.use_diff_chk = QCheckBox("Use motion energy (frame diff)")
        self.use_diff_chk.setChecked(True)

        self.win_spin = QDoubleSpinBox()
        self.win_spin.setRange(0.1, 30.0)
        self.win_spin.setSingleStep(0.1)
        self.win_spin.setValue(2.0)
        self.win_spin.setSuffix(" s")

        self.hop_spin = QDoubleSpinBox()
        self.hop_spin.setRange(0.01, 10.0)
        self.hop_spin.setSingleStep(0.01)
        self.hop_spin.setValue(0.25)
        self.hop_spin.setSuffix(" s")

        grid = QGridLayout()
        grid.addWidget(open_btn,              0, 0, 1, 2)
        grid.addWidget(QLabel("Frame step:"), 1, 0)
        grid.addWidget(self.step_spin,        1, 1)
        grid.addWidget(QLabel("Blur σ:"),     2, 0)
        grid.addWidget(self.blur_spin,        2, 1)
        grid.addWidget(self.use_roi_chk,      3, 0, 1, 2)
        grid.addWidget(self.use_diff_chk,     4, 0, 1, 2)
        grid.addWidget(QLabel("Window (s):"), 5, 0)
        grid.addWidget(self.win_spin,         5, 1)
        grid.addWidget(QLabel("Hop (s):"),    6, 0)
        grid.addWidget(self.hop_spin,         6, 1)
        grid.addWidget(analyze_btn,           7, 0, 1, 2)
        grid.addWidget(export_btn,            8, 0, 1, 2)
        controls_box.setLayout(grid)

        left = QVBoxLayout()
        left.addWidget(self.preview)
        left.addWidget(controls_box)
        left_w = QWidget()
        left_w.setLayout(left)

        # --- Right side: two Qt Charts
        # Top chart: Spectrum
        self.chart_spec = make_chart("Global Amplitude Spectrum", "Frequency (Hz)", "Amplitude")
        self.view_spec = QChartView(self.chart_spec)
        self.view_spec.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.series_spec = None  # recreated per run

        # Bottom chart: Dominant frequency vs time
        self.chart_dom = make_chart("Primary Temporal Frequency vs Time", "Time (s)", "Dominant Freq (Hz)")
        self.view_dom = QChartView(self.chart_dom)
        self.view_dom.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.series_dom_line = None
        self.series_dom_points = None

        right = QVBoxLayout()
        right.addWidget(self.view_spec, stretch=1)
        right.addWidget(self.view_dom,  stretch=1)
        right_w = QWidget()
        right_w.setLayout(right)

        # Main layout
        main = QWidget()
        layout = QHBoxLayout(main)
        layout.addWidget(left_w, 5)
        layout.addWidget(right_w, 6)
        self.setCentralWidget(main)

        # Menu
        file_menu = self.menuBar().addMenu("&File")
        act_open = QAction("Open…", self)
        act_open.triggered.connect(self.open_video)
        file_menu.addAction(act_open)
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

    def on_roi_changed(self, rect: QRect):
        self.current_roi = rect
        if not rect.isNull():
            self.status.showMessage(f"ROI set: x={rect.x()} y={rect.y()} w={rect.width()} h={rect.height()}")
        else:
            self.status.showMessage("ROI cleared")

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.mov *.mkv *.avi);;All Files (*.*)"
        )
        if not path:
            return
        self.video_path = path
        self.setWindowTitle(f"Video Temporal FFT Analyzer — {os.path.basename(path)}")
        self.status.showMessage("Loading preview…")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open video.")
            return

        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            QMessageBox.critical(self, "Error", "Failed to read first frame.")
            return

        self.preview_frame = frame
        self.preview.setPixmap(cvimg_to_qpixmap(self.preview_frame))
        self.status.showMessage("Preview loaded.")

    def run_fft(self):
        if not self.video_path:
            QMessageBox.information(self, "No video", "Open a video first.")
            return

        roi = self.current_roi if self.use_roi_chk.isChecked() else None
        if self.use_roi_chk.isChecked() and (roi is None or roi.isNull()):
            QMessageBox.information(self, "ROI needed", "ROI is enabled but empty. Drag a rectangle on the preview or uncheck ROI.")
            return

        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "Analysis already running.")
            return

        # Reset charts
        self.chart_spec.removeAllSeries()
        self.series_spec = None
        self.chart_dom.removeAllSeries()
        self.series_dom_line = None
        self.series_dom_points = None
        self.last_result = None
        self.status.showMessage("Starting FFT…")

        self.worker = VideoFFTWorker(
            self.video_path,
            frame_step=self.step_spin.value(),
            blur_sigma=self.blur_spin.value(),
            roi_rect=roi,
            use_diff=self.use_diff_chk.isChecked(),
            win_sec=self.win_spin.value(),
            hop_sec=self.hop_spin.value()
        )
        self.worker.progress.connect(self.status.showMessage)
        self.worker.error.connect(self.on_worker_error)
        self.worker.result.connect(self.on_worker_result)
        self.worker.start()

    def on_worker_error(self, msg):
        self.status.showMessage("")
        QMessageBox.critical(self, "Error", msg)

    # -------- Plot helpers (Qt Charts) --------
    def _plot_spectrum(self, freqs, amp):
        freqs = np.asarray(freqs); amp = np.asarray(amp)
        if freqs.size > 1:   # drop DC
            freqs = freqs[1:]
            amp = amp[1:]
        m = np.isfinite(freqs) & np.isfinite(amp)
        freqs = freqs[m]; amp = amp[m]

        self.series_spec = QLineSeries()
        for x, y in zip(freqs.tolist(), amp.tolist()):
            self.series_spec.append(float(x), float(y))
        self.chart_spec.addSeries(self.series_spec)

        # attach to axes
        ax_x = next((ax for ax in self.chart_spec.axes(Qt.AlignmentFlag.AlignBottom)), None)
        ax_y = next((ax for ax in self.chart_spec.axes(Qt.AlignmentFlag.AlignLeft)), None)
        if ax_x is None or ax_y is None:
            # If not present (should be), add them
            self.chart_spec.removeAxis(ax_x) if ax_x else None
            self.chart_spec.removeAxis(ax_y) if ax_y else None
            ax_x = QValueAxis(); ax_y = QValueAxis()
            ax_x.setTitleText("Frequency (Hz)")
            ax_y.setTitleText("Amplitude")
            ax_x.setLabelFormat("%.3f"); ax_y.setLabelFormat("%.3f")
            self.chart_spec.addAxis(ax_x, Qt.AlignmentFlag.AlignBottom)
            self.chart_spec.addAxis(ax_y, Qt.AlignmentFlag.AlignLeft)

        self.series_spec.attachAxis(ax_x)
        self.series_spec.attachAxis(ax_y)

        if freqs.size:
            x_max = max(1.0, float(np.max(freqs)))
        else:
            x_max = 1.0
        if amp.size:
            y_min = float(np.min(amp))
            y_max = float(np.max(amp))
            if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
                y_min, y_max = -1e-6, 1e-6
        else:
            y_min, y_max = -1e-6, 1e-6

        set_xy_limits(self.chart_spec, 0.0, x_max, y_min, y_max * (1.05 if y_max > 0 else 1.0))

    def _plot_dominant_vs_time(self, times, freqs_hz, total_dur):
        times = np.asarray(times); freqs_hz = np.asarray(freqs_hz)
        m = np.isfinite(times) & np.isfinite(freqs_hz)
        t = times[m]; f = freqs_hz[m]

        # Line series (for continuity) + (optional) point series (reuse line only)
        self.series_dom_line = QLineSeries()
        for x, y in zip(t.tolist(), f.tolist()):
            self.series_dom_line.append(float(x), float(y))
        self.chart_dom.addSeries(self.series_dom_line)

        # Attach axes (create if missing)
        ax_x = next((ax for ax in self.chart_dom.axes(Qt.AlignmentFlag.AlignBottom)), None)
        ax_y = next((ax for ax in self.chart_dom.axes(Qt.AlignmentFlag.AlignLeft)), None)
        if ax_x is None or ax_y is None:
            self.chart_dom.removeAxis(ax_x) if ax_x else None
            self.chart_dom.removeAxis(ax_y) if ax_y else None
            ax_x = QValueAxis(); ax_y = QValueAxis()
            ax_x.setTitleText("Time (s)")
            ax_y.setTitleText("Dominant Freq (Hz)")
            ax_x.setLabelFormat("%.2f"); ax_y.setLabelFormat("%.2f")
            self.chart_dom.addAxis(ax_x, Qt.AlignmentFlag.AlignBottom)
            self.chart_dom.addAxis(ax_y, Qt.AlignmentFlag.AlignLeft)

        self.series_dom_line.attachAxis(ax_x)
        self.series_dom_line.attachAxis(ax_y)

        x_max = float(total_dur) if np.isfinite(total_dur) else (float(np.max(t)) if t.size else 1.0)
        if f.size:
            y0 = float(np.min(f))
            y1 = float(np.max(f))
            if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
                y0, y1 = 0.0, 1.0
        else:
            y0, y1 = 0.0, 1.0

        set_xy_limits(self.chart_dom, 0.0, max(1e-3, x_max), y0, y1 * (1.05 if y1 > 0 else 1.0))

    def on_worker_result(self, res: dict):
        self.last_result = res
        self._plot_spectrum(res["freqs"], res["amplitude"])
        self._plot_dominant_vs_time(res["dom_times"], res["dom_freqs"], res.get("duration_s", float('nan')))
        self.status.showMessage(
            f"Done. Samples: {res['n_samples']} | fps: {res['fps']:.3f} | eff fps: {res['effective_fps']:.3f} | "
            f"window: {self.win_spin.value():.2f}s hop: {self.hop_spin.value():.2f}s"
        )

    def export_csv(self):
        if not self.last_result:
            QMessageBox.information(self, "No data", "Run an analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Spectrum CSV", "spectrum.csv", "CSV (*.csv)")
        if not path:
            return
        freqs = self.last_result["freqs"]
        amp = self.last_result["amplitude"]
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frequency_hz", "amplitude"])
                for fz, a in zip(freqs, amp):
                    w.writerow([fz, a])
            self.status.showMessage(f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to write CSV: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
