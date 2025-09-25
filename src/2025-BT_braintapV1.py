import threading
import time
import queue
from datetime import datetime
import re
import tkinter as tk
from tkinter import ttk, messagebox
import serial
from serial.tools import list_ports

APP_TITLE = "ESP32 Bluetooth (SPP) Tester — v4"
DEFAULT_BAUD = 115200  # SPP ignores baud but pyserial requires a value

# ---------- Regex parsers ----------
RE_A0   = re.compile(r"\bA0\s*=\s*(\d+)", re.I)
RE_A1   = re.compile(r"\bA1\s*=\s*(\d+)", re.I)
RE_D1   = re.compile(r"\bD1\s*=\s*([01])", re.I)
RE_D2   = re.compile(r"\bD2\s*=\s*([01])", re.I)
RE_BTN  = re.compile(r"\bBTN\s*[:=]\s*([01])", re.I)
RE_FREQ = re.compile(r"\bFREQ\s*=?\s*(\d+)\s*HZ", re.I)
RE_SRC  = re.compile(r"\bSRC\s*=\s*(ANALOG|MANUAL)", re.I)
RE_DAC_GET  = re.compile(r"\bDAC\s*=\s*(\d+)\s*/\s*255", re.I)
RE_PWM_GET  = re.compile(r"\bPWM\s*=\s*(\d+)\s*/\s*4095", re.I)
RE_DAC_ACK_8   = re.compile(r"\bdac8\s*=\s*(\d+)", re.I)
RE_PWM_ACK     = re.compile(r"\bSET_PWM\s+duty\s*=\s*(\d+)", re.I)

class SerialWorker(threading.Thread):
    def __init__(self, ser: serial.Serial, out_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.ser = ser
        self.out_queue = out_queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                line = self.ser.readline()
                if line:
                    try:
                        text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                    except Exception:
                        text = repr(line)
                    self.out_queue.put(text)
            except serial.SerialException as e:
                self.out_queue.put(f"!! Serial error: {e}")
                break
            except Exception as e:
                self.out_queue.put(f"!! Unexpected error: {e}")
                time.sleep(0.2)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1000x760")
        self.minsize(900, 680)

        self.ser = None
        self.reader_thread = None
        self.stop_event = threading.Event()
        self.rx_queue = queue.Queue()

        # status vars
        self.a0_var = tk.StringVar(value="?")
        self.a1_var = tk.StringVar(value="?")
        self.d1_var = tk.StringVar(value="?")
        self.d2_var = tk.StringVar(value="?")
        self.btn_var = tk.StringVar(value="?")
        self.freq_var_disp = tk.StringVar(value="?")
        self.src_var = tk.StringVar(value="?")
        self.dac_var = tk.StringVar(value="?")   # shows 0..255 (actual DAC8 reported by device)
        self.pwm_var = tk.StringVar(value="?")   # shows 0..4095

        # send controls state
        self.freq_send_var = tk.StringVar(value="100")
        self.dac_send_var = tk.StringVar(value="0")
        self.pwm_send_var = tk.StringVar(value="0")

        # DAC slider throttle
        self._dac_slider_after_id = None
        self._dac_last_sent = -1

        self._build_ui()
        self._refresh_ports()
        self._poll_rx_queue()

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Port:").grid(row=0, column=0, sticky="w")
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(top, textvariable=self.port_var, state="readonly", width=36)
        self.port_combo.grid(row=0, column=1, sticky="w", padx=(5, 10))

        self.refresh_btn = ttk.Button(top, text="Refresh", command=self._refresh_ports)
        self.refresh_btn.grid(row=0, column=2, sticky="w")

        self.connect_btn = ttk.Button(top, text="Connect", command=self._toggle_connection)
        self.connect_btn.grid(row=0, column=3, sticky="w", padx=(10, 0))

        self.status_var = tk.StringVar(value="Disconnected")
        ttk.Label(top, textvariable=self.status_var).grid(row=0, column=4, sticky="w", padx=(12, 0))

        # COMMANDS
        cmds = ttk.LabelFrame(self, text="Commands", padding=(10, 6))
        cmds.pack(fill="x", padx=10, pady=(6, 0))

        row0 = ttk.Frame(cmds); row0.pack(fill="x", pady=4)
        ttk.Button(row0, text="LED ON",  command=lambda: self._send_cmd("LED ON")).pack(side="left", padx=(0,6))
        ttk.Button(row0, text="LED OFF", command=lambda: self._send_cmd("LED OFF")).pack(side="left", padx=(0,6))
        ttk.Button(row0, text="READ BTN", command=lambda: self._send_cmd("READ")).pack(side="left", padx=(0,6))
        ttk.Button(row0, text="READALL", command=lambda: self._send_cmd("READALL")).pack(side="left", padx=(0,6))
        ttk.Button(row0, text="GET_FREQ", command=lambda: self._send_cmd("GET_FREQ")).pack(side="left", padx=(0,6))
        ttk.Button(row0, text="GET_ANALOG", command=lambda: self._send_cmd("GET_ANALOG")).pack(side="left", padx=(0,6))
        ttk.Button(row0, text="FREQ SRC ANALOG", command=lambda: self._send_cmd("FREQ SRC ANALOG")).pack(side="left", padx=(12,6))
        ttk.Button(row0, text="FREQ SRC MANUAL", command=lambda: self._send_cmd("FREQ SRC MANUAL")).pack(side="left", padx=(0,6))
        ttk.Button(row0, text="Clear Log", command=self._clear_log).pack(side="right")

        # Row 1: UPDATE_FREQ
        row1 = ttk.Frame(cmds); row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Update Freq (Hz 1–2000):").pack(side="left")
        self.freq_spin = tk.Spinbox(
            row1, from_=1, to=2000, increment=1,
            textvariable=self.freq_send_var, width=8,
            validate="key",
            validatecommand=(self.register(lambda s: s.isdigit() or s == ""), "%P"),
        )
        self.freq_spin.pack(side="left", padx=(6,6))
        ttk.Button(row1, text="Send UPDATE_FREQ", command=self._send_update_freq).pack(side="left")
        self.freq_spin.bind("<Return>", lambda e: self._send_update_freq())

        # Row 2: DAC slider (auto-send)
        row2 = ttk.LabelFrame(cmds, text="DAC control (0–4095 raw → device scales to 0–255)", padding=(10, 8))
        row2.pack(fill="x", pady=6)
        self.dac_scale = tk.Scale(
            row2, from_=0, to=4095, orient="horizontal", showvalue=False,
            length=700, command=self._on_dac_slider_move, resolution=1
        )
        self.dac_scale.pack(fill="x", padx=6)
        dac_controls = ttk.Frame(row2); dac_controls.pack(fill="x", pady=(6,0))
        self.dac_value_label = ttk.Label(dac_controls, text="0")
        self.dac_value_label.pack(side="left")
        ttk.Button(dac_controls, text="Send SET_DAC (current)", command=self._send_set_dac_current).pack(side="left", padx=(10,0))
        # Keep the old spinbox as an alternative (optional). Comment out if you don't want it:
        ttk.Label(dac_controls, text=" / or type:").pack(side="left", padx=(10,4))
        self.dac_spin = tk.Spinbox(
            dac_controls, from_=0, to=4095, increment=1,
            textvariable=self.dac_send_var, width=8,
            validate="key",
            validatecommand=(self.register(lambda s: s.isdigit() or s == ""), "%P"),
        )
        self.dac_spin.pack(side="left", padx=(0,6))
        ttk.Button(dac_controls, text="Send", command=self._send_set_dac).pack(side="left")
        self.dac_spin.bind("<Return>", lambda e: self._send_set_dac())

        # Row 3: SET_PWM (0..4095)
        row3 = ttk.Frame(cmds); row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="PWM (0–4095):").pack(side="left")
        self.pwm_spin = tk.Spinbox(
            row3, from_=0, to=4095, increment=1,
            textvariable=self.pwm_send_var, width=8,
            validate="key",
            validatecommand=(self.register(lambda s: s.isdigit() or s == ""), "%P"),
        )
        self.pwm_spin.pack(side="left", padx=(6,6))
        ttk.Button(row3, text="Send SET_PWM", command=self._send_set_pwm).pack(side="left")
        self.pwm_spin.bind("<Return>", lambda e: self._send_set_pwm())

        # LOG
        logframe = ttk.Frame(self, padding=(10, 6, 10, 10))
        logframe.pack(fill="both", expand=True)
        self.log = tk.Text(logframe, wrap="none", height=18, undo=False)
        self.log.configure(state="disabled")
        yscroll = ttk.Scrollbar(logframe, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=yscroll.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        logframe.rowconfigure(0, weight=1)
        logframe.columnconfigure(0, weight=1)

        # SEND BAR
        sendbar = ttk.Frame(self, padding=(10, 0, 10, 6))
        sendbar.pack(fill="x")
        ttk.Label(sendbar, text="Send raw:").pack(side="left")
        self.send_var = tk.StringVar()
        self.send_entry = ttk.Entry(sendbar, textvariable=self.send_var)
        self.send_entry.pack(side="left", fill="x", expand=True, padx=(6, 6))
        self.send_entry.bind("<Return>", lambda e: self._on_send_clicked())
        self.append_nl_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sendbar, text="Append \\n", variable=self.append_nl_var).pack(side="left")
        ttk.Button(sendbar, text="Send", command=self._on_send_clicked).pack(side="left", padx=(8, 0))

        # CURRENT DEVICE STATUS
        status = ttk.LabelFrame(self, text="Current device status", padding=(10, 8))
        status.pack(fill="x", padx=10, pady=(4, 12))

        def add_row(parent, label, var):
            row = ttk.Frame(parent)
            row.pack(side="left", padx=(0, 20))
            ttk.Label(row, text=label).pack(side="top", anchor="w")
            ttk.Label(row, textvariable=var, font=("TkDefaultFont", 12, "bold")).pack(side="top")

        add_row(status, "A0:", self.a0_var)
        add_row(status, "A1:", self.a1_var)
        add_row(status, "D1:", self.d1_var)
        add_row(status, "D2:", self.d2_var)
        add_row(status, "BTN:", self.btn_var)
        add_row(status, "FREQ (Hz):", self.freq_var_disp)
        add_row(status, "SRC:", self.src_var)
        add_row(status, "DAC (0–255):", self.dac_var)
        add_row(status, "PWM (0–4095):", self.pwm_var)

        # style tweak
        style = ttk.Style(self)
        try:
            self.call('tk', 'scaling', 1.0)
        except Exception:
            pass

    # ---------- Serial handling ----------
    def _refresh_ports(self):
        ports = list_ports.comports()
        items = [f"{p.device} - {p.description}" for p in ports]
        self.port_combo['values'] = items
        if items and not self.port_var.get():
            self.port_var.set(items[0])

    def _get_selected_port(self):
        val = self.port_var.get().strip()
        return val.split(" ")[0] if val else ""

    def _toggle_connection(self):
        if self.ser and self.ser.is_open:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self._get_selected_port()
        if not port:
            messagebox.showwarning("No port", "Pick a Bluetooth COM port first.")
            return
        try:
            self.ser = serial.Serial(port, DEFAULT_BAUD, timeout=1)
        except Exception as e:
            messagebox.showerror("Connection failed", f"Could not open {port}\n\n{e}")
            return
        self.status_var.set(f"Connected: {port}")
        self.connect_btn.configure(text="Disconnect")
        self.stop_event.clear()
        self.reader_thread = SerialWorker(self.ser, self.rx_queue, self.stop_event)
        self.reader_thread.start()
        self._log(f"[Connected to {port}]")

    def _disconnect(self):
        self._log("[Disconnecting…]")
        self.status_var.set("Disconnecting…")
        try:
            self.stop_event.set()
            if self.reader_thread and self.reader_thread.is_alive():
                self.reader_thread.join(timeout=1.5)
        except Exception:
            pass
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None
        self.reader_thread = None
        self.stop_event.clear()
        self.status_var.set("Disconnected")
        self.connect_btn.configure(text="Connect")
        self._log("[Disconnected]")

    # ---------- Send helpers ----------
    def _on_send_clicked(self):
        text = self.send_var.get().strip()
        if not text:
            return
        self._send_text(text)
        self.send_var.set("")

    def _send_text(self, text: str):
        if not (self.ser and self.ser.is_open):
            self._log("!! Not connected")
            return
        try:
            payload = text + ("\n" if self.append_nl_var.get() else "")
            self.ser.write(payload.encode("utf-8"))
            self._log(f">> {text}")
        except Exception as e:
            self._log(f"!! Send failed: {e}")

    def _send_cmd(self, cmd: str):
        self._send_text(cmd)

    def _send_update_freq(self):
        raw = (self.freq_send_var.get() or "").strip()
        if not raw.isdigit():
            messagebox.showwarning("Invalid value", "Enter a number from 1 to 2000.")
            return
        val = int(raw)
        if not (1 <= val <= 2000):
            messagebox.showwarning("Out of range", "Value must be between 1 and 2000.")
            return
        self._send_text(f"UPDATE_FREQ {val}")

    def _send_set_dac_current(self):
        val = int(float(self.dac_scale.get()))
        self._send_text(f"SET_DAC {val}")

    def _send_set_dac(self):
        raw = (self.dac_send_var.get() or "").strip()
        if not raw.isdigit():
            messagebox.showwarning("Invalid value", "Enter a number from 0 to 4095.")
            return
        val = int(raw)
        if not (0 <= val <= 4095):
            messagebox.showwarning("Out of range", "Value must be between 0 and 4095.")
            return
        # also sync slider to this value
        try: self.dac_scale.set(val)
        except Exception: pass
        self._send_text(f"SET_DAC {val}")

    def _send_set_pwm(self):
        raw = (self.pwm_send_var.get() or "").strip()
        if not raw.isdigit():
            messagebox.showwarning("Invalid value", "Enter a number from 0 to 4095.")
            return
        val = int(raw)
        if not (0 <= val <= 4095):
            messagebox.showwarning("Out of range", "Value must be between 0 and 4095.")
            return
        self._send_text(f"SET_PWM {val}")

    # ---------- DAC slider logic (auto-send with light rate limit) ----------
    def _on_dac_slider_move(self, _):
        val = int(float(self.dac_scale.get()))
        self.dac_value_label.configure(text=str(val))
        # throttle rapid sends: schedule after 25ms; reschedule if still moving
        if self._dac_slider_after_id is not None:
            try:
                self.after_cancel(self._dac_slider_after_id)
            except Exception:
                pass
        self._dac_slider_after_id = self.after(25, self._send_dac_if_changed, val)

    def _send_dac_if_changed(self, val):
        self._dac_slider_after_id = None
        if val != self._dac_last_sent:
            self._dac_last_sent = val
            self._send_text(f"SET_DAC {val}")

    # ---------- RX queue & parsing ----------
    def _poll_rx_queue(self):
        try:
            while True:
                line = self.rx_queue.get_nowait()
                self._log(f"<< {line}")
                self._parse_status_from_line(line)
        except queue.Empty:
            pass
        self.after(16, self._poll_rx_queue)

    def _parse_status_from_line(self, line: str):
        m = RE_A0.search(line)
        if m:
            self.a0_var.set(m.group(1))
        m = RE_A1.search(line)
        if m:
            self.a1_var.set(m.group(1))
        m = RE_D1.search(line)
        if m:
            self.d1_var.set(m.group(1))
        m = RE_D2.search(line)
        if m:
            self.d2_var.set(m.group(1))
        m = RE_BTN.search(line)
        if m:
            self.btn_var.set(m.group(1))
        m = RE_FREQ.search(line)
        if m:
            self.freq_var_disp.set(m.group(1))
        m = RE_SRC.search(line)
        if m:
            self.src_var.set(m.group(1).upper())

        m = RE_DAC_GET.search(line)
        if m:
            self.dac_var.set(m.group(1))
        m = RE_PWM_GET.search(line)
        if m:
            self.pwm_var.set(m.group(1))
        m = RE_DAC_ACK_8.search(line)
        if m:
            self.dac_var.set(m.group(1))
        m = RE_PWM_ACK.search(line)
        if m:
            self.pwm_var.set(m.group(1))

    # ---------- Logging ----------
    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log.configure(state="normal")
        self.log.insert("end", line)
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---------- Clean up ----------
    def destroy(self):
        try:
            if self.ser and self.ser.is_open:
                self._disconnect()
        finally:
            super().destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
