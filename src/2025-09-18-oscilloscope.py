# File: esp32_8ch_scope_gui_pyqt6_plus_freq.py
import struct
import sys
import time
import threading
from collections import deque

import numpy as np
import serial
import serial.tools.list_ports

from PyQt6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

# -------------------- Protocol --------------------
MAGIC = b'OS'
HDR_FMT = '<2sBBHII'   # magic[2], ver u8, ch u8, n_sets u16, total_sps u32, seq u32
HDR_SZ  = struct.calcsize(HDR_FMT)

# -------------------- ADC / Voltage conversion --------------------
# ESP32 (classic) nominal full-scale for atten steps (approximate)
# These are the typical spans Espressif documents; real devices vary.
ESP32_ATTEN_TO_FS = {
    "0 dB  (~1.1 V FS)": 1.1,
    "2.5 dB (~1.5 V FS)": 1.5,
    "6 dB  (~2.2 V FS)": 2.2,
    "11 dB (~3.3 V FS)": 3.3,
}
ADC_BITS = 12
ADC_FULL_SCALE_CODE = (1 << ADC_BITS) - 1  # 4095

def adc_to_volts(adc_u16, fs_volts):
    """
    Convert 12-bit ADC code (0..4095) to volts given selected full-scale.
    Our stream is 12-bit left-aligned in 16 bits; we right-shift by 4 before using.
    """
    codes = (adc_u16 >> 4).astype(np.float32)  # back to 12b code
    return (codes / ADC_FULL_SCALE_CODE) * fs_volts

# -------------------- Serial Reader --------------------
class SerialReader(QtCore.QObject):
    frame_received = QtCore.pyqtSignal(np.ndarray, int, int, int)  # arr[n_sets, ch], n_sets, ch, total_rate
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.ser = None
        self._stop = threading.Event()
        self._thread = None

    def open(self, port, baud):
        try:
            # Open without resetting ESP32 (hold DTR/RTS low)
            self.ser = serial.Serial()
            self.ser.port = port
            self.ser.baudrate = baud
            self.ser.timeout = 1
            self.ser.dtr = False
            self.ser.rts = False
            self.ser.open()

            # Let MCU settle and flush any boot junk
            time.sleep(0.3)
            self.ser.reset_input_buffer()

            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self.status.emit(f"Opened {port} @ {baud} (no reset)")
        except Exception as e:
            self.error.emit(f"Open failed: {e}")

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.ser:
            try:
                self.ser.close()
            except:
                pass
        self.ser = None
        self.status.emit("Closed")

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n and not self._stop.is_set():
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                continue
            buf.extend(chunk)
        return bytes(buf) if len(buf) == n else None

    def _sync_to_magic(self):
        win = bytearray(2)
        idle_start = time.time()
        while not self._stop.is_set():
            b = self.ser.read(1)
            if not b:
                if time.time() - idle_start > 1.0:
                    self.ser.reset_input_buffer()
                    idle_start = time.time()
                continue
            win[0] = win[1]
            win[1] = b[0]
            if bytes(win) == MAGIC:
                return True
        return False

    def _run(self):
        try:
            while not self._stop.is_set():
                if not self._sync_to_magic():
                    break
                rest = self._read_exact(HDR_SZ - 2)
                if not rest:
                    continue
                magic, ver, channels, n_sets, samp_rate_total, seq = struct.unpack(HDR_FMT, MAGIC + rest)
                if magic != MAGIC or channels == 0 or n_sets == 0:
                    continue
                sample_count = n_sets * channels
                payload = self._read_exact(sample_count * 2)
                if payload is None:
                    self.status.emit("Timeout on payload; resyncing…")
                    continue
                arr = np.frombuffer(payload, dtype='<u2').reshape((n_sets, channels))
                self.frame_received.emit(arr, n_sets, channels, samp_rate_total)
        except Exception as e:
            self.error.emit(f"Reader error: {e}")

# -------------------- Main GUI --------------------
class ScopePlot(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32 8-Channel Scope (PyQt6) — with Voltage/ADC & Frequency Panel")

        # ---- Top bar (port) ----
        self.port_cb = QtWidgets.QComboBox()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.baud_cb = QtWidgets.QComboBox()
        self.baud_cb.addItems([str(b) for b in (2000000, 3000000, 921600, 115200)])
        self.open_btn = QtWidgets.QPushButton("Open")
        self.close_btn = QtWidgets.QPushButton("Close")
        self.status_lbl = QtWidgets.QLabel("Idle")
        self.fps_lbl = QtWidgets.QLabel("-- fps")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.port_cb)
        top.addWidget(self.refresh_btn)
        top.addWidget(QtWidgets.QLabel("Baud:"))
        top.addWidget(self.baud_cb)
        top.addWidget(self.open_btn)
        top.addWidget(self.close_btn)
        top.addStretch(1)
        top.addWidget(self.fps_lbl)
        top.addWidget(self.status_lbl)

        # ---- Channel toggles & unit controls ----
        ch_layout = QtWidgets.QHBoxLayout()
        self.ch_checks = []
        for i in range(8):
            cb = QtWidgets.QCheckBox(f"CH{i}")
            cb.setChecked(True)
            self.ch_checks.append(cb)
            ch_layout.addWidget(cb)

        self.units_toggle = QtWidgets.QCheckBox("Show Volts (Y)")
        self.units_toggle.setChecked(False)

        self.atten_cb = QtWidgets.QComboBox()
        for k in ESP32_ATTEN_TO_FS.keys():
            self.atten_cb.addItem(k)
        self.atten_cb.setCurrentText("11 dB (~3.3 V FS)")

        self.vref_spin = QtWidgets.QDoubleSpinBox()
        self.vref_spin.setDecimals(3)
        self.vref_spin.setRange(0.8, 4.0)
        self.vref_spin.setSingleStep(0.05)
        self.vref_spin.setValue(1.100)  # nominal internal reference (approx) — used only to annotate

        self.autoscale_btn = QtWidgets.QPushButton("Autoscale")
        self.clear_btn = QtWidgets.QPushButton("Clear")

        units_row = QtWidgets.QHBoxLayout()
        units_row.addLayout(ch_layout)
        units_row.addStretch(1)
        units_row.addWidget(QtWidgets.QLabel("Atten:"))
        units_row.addWidget(self.atten_cb)
        units_row.addWidget(QtWidgets.QLabel("Vref (note):"))
        units_row.addWidget(self.vref_spin)
        units_row.addWidget(self.units_toggle)
        units_row.addWidget(self.autoscale_btn)
        units_row.addWidget(self.clear_btn)

        # ---- Main scope plot ----
        self.scope_plot = pg.PlotWidget(background='k')
        self.scope_plot.addLegend(offset=(10, 10))
        self.scope_plot.showGrid(x=True, y=True, alpha=0.3)
        self.scope_plot.setLabel('bottom', 'Time', units='s')
        self.scope_plot.setLabel('left', 'ADC Code (12-bit)')
        self.curves = [self.scope_plot.plot(pen=pg.intColor(i), name=f"CH{i}") for i in range(8)]

        # ---- Frequency panel controls ----
        self.freq_enable = QtWidgets.QCheckBox("Enable Frequency Panel")
        self.freq_enable.setChecked(True)

        self.fft_source_cb = QtWidgets.QComboBox()
        self.fft_source_cb.addItem("Sum of selected")
        for i in range(8):
            self.fft_source_cb.addItem(f"CH{i}")

        self.nfft_cb = QtWidgets.QComboBox()
        self.nfft_cb.addItems(["512", "1024", "2048"])
        self.nfft_cb.setCurrentText("1024")

        self.overlap_cb = QtWidgets.QComboBox()
        self.overlap_cb.addItems(["0%", "25%", "50%"])
        self.overlap_cb.setCurrentText("0%")

        self.fr_log_cb = QtWidgets.QCheckBox("Log amplitude (dB)")
        self.fr_log_cb.setChecked(False)
        self.fr_autoscale_cb = QtWidgets.QCheckBox("Auto-scale spectrum")
        self.fr_autoscale_cb.setChecked(True)

        freq_ctl = QtWidgets.QHBoxLayout()
        freq_ctl.addWidget(self.freq_enable)
        freq_ctl.addStretch(1)
        freq_ctl.addWidget(QtWidgets.QLabel("FFT source:"))
        freq_ctl.addWidget(self.fft_source_cb)
        freq_ctl.addWidget(QtWidgets.QLabel("NFFT:"))
        freq_ctl.addWidget(self.nfft_cb)
        freq_ctl.addWidget(QtWidgets.QLabel("Overlap:"))
        freq_ctl.addWidget(self.overlap_cb)
        freq_ctl.addWidget(self.fr_log_cb)
        freq_ctl.addWidget(self.fr_autoscale_cb)




        # ---- Bottom-left: Spectrogram (ImageItem) ----
        self.spec_plot = pg.PlotWidget(background='k')
        self.spec_plot.showGrid(x=False, y=True, alpha=0.2)
        self.spec_plot.setLabel('left', 'Time', units='s')
        self.spec_plot.setLabel('bottom', 'Frequency', units='Hz')
        # self.spec_plot.setLabel('left', 'Frequency', units='Hz')
        # self.spec_plot.setLabel('bottom', 'Time', units='s')
        self.spec_img = pg.ImageItem()
        self.spec_plot.addItem(self.spec_img)
        self.spec_img.setAutoDownsample(True)

        # ---- Bottom-right: Frequency response (horizontal amplitude, vertical frequency) ----
        self.fr_plot = pg.PlotWidget(background='k')
        self.fr_plot.showGrid(x=True, y=True, alpha=0.2)
        self.fr_plot.setLabel('bottom', 'Amplitude')
        self.fr_plot.setLabel('left', 'Frequency', units='Hz')
        self.fr_curve = self.fr_plot.plot(pen=pg.mkPen('#00ccff', width=2), name="Spectrum")

        # ---- Layout: main scope on top; bottom has two columns ----
        bottom_split = QtWidgets.QHBoxLayout()
        bottom_split.addWidget(self.spec_plot, 3)
        bottom_split.addWidget(self.fr_plot, 2)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(units_row)
        layout.addWidget(self.scope_plot, 3)
        layout.addLayout(freq_ctl)
        layout.addLayout(bottom_split, 2)

        # ---- Data buffers ----
        self.enabled = True
        self.max_points = 6000
        self.buffers = [deque(maxlen=self.max_points) for _ in range(8)]
        self.x = deque(maxlen=self.max_points)
        self.sample_period = 1 / (160000.0 / 8.0)

        # Serial reader
        self.reader = SerialReader()
        self.reader.frame_received.connect(self.on_frame)
        self.reader.status.connect(self.on_status)
        self.reader.error.connect(self.on_error)

        # Signals
        self.refresh_btn.clicked.connect(self.refresh_ports)
        self.open_btn.clicked.connect(self.open_port)
        self.close_btn.clicked.connect(self.close_port)
        self.autoscale_btn.clicked.connect(self.autoscale)
        self.clear_btn.clicked.connect(self.clear_data)
        self.units_toggle.stateChanged.connect(self._update_units_label)

        # Plot timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(30)

        self.refresh_ports()

        # ---- Spectrogram state ----
        self.spec_history_secs = 10.0          # how many seconds to keep horizontally
        self._spec_img_array = None            # 2D array [freq_bins, time_cols]
        self._spec_time_axis = deque(maxlen=10000)
        self._last_fft_tail = None             # for overlap
        self._last_spec_time = None
        self._last_stats_t = time.time()
        self._frames_since = 0

        # Colormap & levels for spectrogram
        lut = pg.colormap.get('inferno').getLookupTable(alpha=False)
        self.spec_img.setLookupTable(lut)
        self.spec_img.setLevels([ -80, 0 ])    # dBFS-ish

        # Initial units label
        self._update_units_label()

    # ---------- Basic wiring ----------
    @QtCore.pyqtSlot()
    def refresh_ports(self):
        self.port_cb.clear()
        for p in serial.tools.list_ports.comports():
            self.port_cb.addItem(p.device)

    @QtCore.pyqtSlot()
    def open_port(self):
        port = self.port_cb.currentText()
        baud = int(self.baud_cb.currentText())
        if not port:
            self.on_status("No COM port selected")
            return
        self.reader.open(port, baud)

    @QtCore.pyqtSlot()
    def close_port(self):
        self.reader.close()

    @QtCore.pyqtSlot(str)
    def on_status(self, s):
        self.status_lbl.setText(s)

    @QtCore.pyqtSlot(str)
    def on_error(self, s):
        self.status_lbl.setText(s)

    # ---------- Frame ingest ----------
    @QtCore.pyqtSlot(np.ndarray, int, int, int)
    def on_frame(self, arr, n_sets, channels, samp_rate_total):
        if not self.enabled:
            return

        self._frames_since += 1
        per_ch_rate = float(samp_rate_total) / float(channels)
        self.sample_period = 1.0 / per_ch_rate

        # Append samples & time index
        for set_idx in range(n_sets):
            self.x.append(self.x[-1] + 1 if len(self.x) else 0)
            row = arr[set_idx]
            for ch in range(min(channels, 8)):
                self.buffers[ch].append(int(row[ch]))

        # Periodic stats
        now = time.time()
        if now - self._last_stats_t >= 0.5:
            fps = self._frames_since / (now - self._last_stats_t)
            self._frames_since = 0
            self._last_stats_t = now
            mins, maxs = [], []
            for ch in range(8):
                if self.buffers[ch]:
                    a = np.frombuffer(np.array(self.buffers[ch], dtype=np.uint16), dtype=np.uint16)
                    mins.append(int(a.min() >> 4))
                    maxs.append(int(a.max() >> 4))
                else:
                    mins.append(-1); maxs.append(-1)
            self.fps_lbl.setText(f"{fps:.1f} fps")
            self.status_lbl.setText(f"min/max per ch: {list(zip(mins, maxs))}")

        # Update frequency panel from most recent data
        if self.freq_enable.isChecked():
            self._update_frequency(arr, per_ch_rate)

    # ---------- Plots ----------
    @QtCore.pyqtSlot()
    def update_plots(self):
        if not self.x:
            return

        # Time axis (seconds)
        idx = np.arange(len(self.x), dtype=np.float32)
        t = idx * self.sample_period

        # Full-scale volts for conversion
        fs_volts = ESP32_ATTEN_TO_FS[self.atten_cb.currentText()]
        show_volts = self.units_toggle.isChecked()

        any_visible = False
        for ch, curve in enumerate(self.curves):
            visible = self.ch_checks[ch].isChecked() and len(self.buffers[ch]) > 0
            curve.setVisible(visible)
            if not visible:
                continue
            any_visible = True

            yu16 = np.frombuffer(np.array(self.buffers[ch], dtype=np.uint16), dtype=np.uint16)
            if show_volts:
                y = adc_to_volts(yu16, fs_volts)
            else:
                y = (yu16 >> 4).astype(np.float32)

            tt = t[-len(y):] if len(t) >= len(y) else np.arange(len(y), dtype=np.float32) * self.sample_period
            curve.setData(tt, y)

        if any_visible:
            self.scope_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
            self.scope_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

    def _update_units_label(self):
        if self.units_toggle.isChecked():
            self.scope_plot.setLabel('left', 'Voltage', units='V')
        else:
            self.scope_plot.setLabel('left', 'ADC Code (12-bit)')

    @QtCore.pyqtSlot()
    def autoscale(self):
        self.scope_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        self.scope_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

    @QtCore.pyqtSlot()
    def clear_data(self):
        self.x.clear()
        for b in self.buffers:
            b.clear()
        self._spec_img_array = None
        self._last_fft_tail = None
        self._spec_time_axis.clear()
        self.fr_curve.setData([], [])

    # ---------- Frequency processing ----------
    def _get_fft_source_trace(self):
        """
        Returns a 1D np.float32 array of the most recent samples
        from either one channel or the sum of selected channels.
        """
        choice = self.fft_source_cb.currentText()
        # Collect equal-length views
        lens = [len(b) for b in self.buffers]
        L = min(lens) if all(lens) else max(lens)
        if L == 0:
            return None

        if choice == "Sum of selected":
            acc = None
            for ch, b in enumerate(self.buffers):
                if not self.ch_checks[ch].isChecked() or len(b) < L:
                    continue
                arr = np.frombuffer(np.array(b, dtype=np.uint16), dtype=np.uint16)[-L:]
                arr = (arr >> 4).astype(np.float32)
                acc = arr if acc is None else (acc + arr)
            return acc if acc is not None else None
        else:
            ch = int(choice.replace("CH", ""))
            if len(self.buffers[ch]) < L:
                return None
            arr = np.frombuffer(np.array(self.buffers[ch], dtype=np.uint16), dtype=np.uint16)[-L:]
            return (arr >> 4).astype(np.float32)

    def _update_frequency(self, new_arr, per_ch_rate):
        """
        Compute one FFT slice and update:
          - Spectrogram (image rolling to the left)
          - Frequency response (horizontal amplitude vs vertical frequency)
        """
        # Build time-domain vector
        x = self._get_fft_source_trace()
        if x is None or x.size < 32:
            return

        # NFFT
        nfft = int(self.nfft_cb.currentText())
        overlap_sel = self.overlap_cb.currentText()
        hop = nfft
        if overlap_sel == "25%":
            hop = int(nfft * 0.75)
        elif overlap_sel == "50%":
            hop = int(nfft * 0.5)

        # Prepare a contiguous frame with optional overlap using _last_fft_tail
        if self._last_fft_tail is not None and self._last_fft_tail.size > 0:
            needed = max(0, hop - self._last_fft_tail.size)
            if needed > 0:
                if x.size < needed:
                    return
                frame = np.concatenate([self._last_fft_tail, x[-needed:]])
            else:
                frame = self._last_fft_tail[-hop:]
        else:
            if x.size < hop:
                return
            frame = x[-hop:]

        # Save tail for next time (overlap context)
        tail_len = nfft - hop
        if tail_len > 0:
            if x.size < nfft:
                return
            self._last_fft_tail = x[-tail_len:].copy()
        else:
            self._last_fft_tail = None

        # If hop < nfft, we need more samples to form a full window
        if x.size < nfft:
            return
        windowed = x[-nfft:].astype(np.float32) * np.hanning(nfft).astype(np.float32)
        spec = np.fft.rfft(windowed)
        mag = np.abs(spec) + 1e-12  # avoid log(0)
        mag_db = 20.0 * np.log10(mag / (nfft/2.0))  # dB-ish

        # Frequency axis (for both spectrogram and response)
        freqs = np.fft.rfftfreq(nfft, d=1.0/per_ch_rate)

        # -------- Spectrogram (append one new column) --------
        # Map mag_db (len F) into image with F rows. We'll append as a new column (time).
        col = mag_db.astype(np.float32)[:, None]  # shape (F,1)

        if self._spec_img_array is None:
            # Allocate enough columns to cover spec_history_secs
            cols = int(max(50, np.ceil(self.spec_history_secs * (per_ch_rate / hop))))
            self._spec_img_array = np.full((col.shape[0], cols), -120.0, dtype=np.float32)

        # Roll left by 1, insert newest column at the rightmost
        self._spec_img_array = np.roll(self._spec_img_array, -1, axis=1)
        self._spec_img_array[:, -1] = col[:, 0]

        # Update spectrogram image; set axes so Y is frequency
        # Update spectrogram image; set axes so Y is frequency
        self.spec_img.setImage(self._spec_img_array, autoLevels=False)

        cols = self._spec_img_array.shape[1]
        dur = cols * (hop / per_ch_rate)  # total seconds currently visible (X extent)

        # Map image pixels -> data coordinates (seconds, Hz)
        self.spec_img.resetTransform()
        xf = dur / cols
        yf = freqs[-1] / (self._spec_img_array.shape[0] - 1)
        T = QtGui.QTransform()
        T.scale(xf, yf)
        self.spec_img.setTransform(T)

        # Put the newest column at the right edge and scroll left over time
        # (Keep time increasing to the right; oldest at ~0 s, newest at ~dur s)
        self.spec_img.setPos(0, 0)

        # Make the viewbox show the full data rect explicitly
        self.spec_plot.setXRange(0, dur, padding=0)
        self.spec_plot.setYRange(0, freqs[-1], padding=0)
        auto = self.fr_autoscale_cb.isChecked()
        self.spec_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=auto)
        self.spec_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=auto)

        # self.spec_img.resetTransform()
        # self.spec_img.scale(dur / cols, freqs[-1] / (self._spec_img_array.shape[0] - 1))
        # self.spec_img.setPos(-dur, 0)  # newest at right, time flowing left
        # self.spec_img.resetTransform()
        # xf = dur / cols
        # yf = freqs[-1] / (self._spec_img_array.shape[0] - 1)
        # T = QtGui.QTransform()
        # T.scale(xf, yf)
        # self.spec_img.setTransform(T)
        # self.spec_img.setPos(-dur, 0)  # newest column at the right, time flows left

        # -------- Frequency response (horizontal amplitude, vertical frequency) --------
        # Use linear amplitude (or dB). We'll display linear (>=0) to match "amplitude to the right".
        amp = np.abs(spec).astype(np.float32)
        # NEW: choose linear vs dB and adjust label
        if self.fr_log_cb.isChecked():
            xvals = 20.0 * np.log10(np.maximum(amp, 1e-12))
            self.fr_plot.setLabel('bottom', 'Amplitude (dB)')
        else:
            xvals = amp
            self.fr_plot.setLabel('bottom', 'Amplitude')

        self.fr_curve.setData(xvals, freqs)  # x = amplitude, y = frequency

        # NEW: honor autoscale toggle (when off, user can pan/zoom freely)
        auto = self.fr_autoscale_cb.isChecked()
        self.fr_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=auto)
        self.fr_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=auto)

# -------------------- Main --------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = ScopePlot()
    w.resize(1280, 900)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
