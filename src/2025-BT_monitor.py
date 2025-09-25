import threading
import time
import queue
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import serial
from serial.tools import list_ports

APP_TITLE = "RalphTap - BT tester"
DEFAULT_BAUD = 115200  # Baud is mostly ignored by SPP, but pyserial needs a value


class SerialWorker(threading.Thread):
    def __init__(self, ser: serial.Serial, out_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.ser = ser
        self.out_queue = out_queue
        self.stop_event = stop_event

    def run(self):
        # Continuously read lines and push to queue
        while not self.stop_event.is_set():
            try:
                line = self.ser.readline()  # blocks until timeout
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
        self.geometry("820x520")
        self.minsize(740, 460)

        self.ser = None
        self.reader_thread = None
        self.stop_event = threading.Event()
        self.rx_queue = queue.Queue()

        self._build_ui()
        self._refresh_ports()
        self._poll_rx_queue()

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        # Port picker row
        ttk.Label(top, text="Port:").grid(row=0, column=0, sticky="w")
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(top, textvariable=self.port_var, state="readonly", width=28)
        self.port_combo.grid(row=0, column=1, sticky="w", padx=(5, 10))

        self.refresh_btn = ttk.Button(top, text="Refresh", command=self._refresh_ports)
        self.refresh_btn.grid(row=0, column=2, sticky="w")

        self.connect_btn = ttk.Button(top, text="Connect", command=self._toggle_connection)
        self.connect_btn.grid(row=0, column=3, sticky="w", padx=(10, 0))

        self.status_var = tk.StringVar(value="Disconnected")
        ttk.Label(top, textvariable=self.status_var).grid(row=0, column=4, sticky="w", padx=(12, 0))

        # Button state indicator
        self.btn_state_var = tk.StringVar(value="BTN: ?")
        self.btn_state_label = ttk.Label(top, textvariable=self.btn_state_var)
        self.btn_state_label.grid(row=0, column=5, sticky="e", padx=(30, 0))

        # Quick command buttons
        cmdbar = ttk.Frame(self, padding=(10, 0, 10, 6))
        cmdbar.pack(fill="x")
        ttk.Button(cmdbar, text="LED ON", command=lambda: self._send_text("LED ON")).pack(side="left")
        ttk.Button(cmdbar, text="LED OFF", command=lambda: self._send_text("LED OFF")).pack(side="left", padx=(6, 0))
        ttk.Button(cmdbar, text="READ BTN", command=lambda: self._send_text("READ")).pack(side="left", padx=(6, 0))
        ttk.Button(cmdbar, text="Clear Log", command=self._clear_log).pack(side="left", padx=(18, 0))

        # --- NEW: Update freq controls ---
        freqbar = ttk.Frame(self, padding=(10, 0, 10, 6))
        freqbar.pack(fill="x")
        ttk.Label(freqbar, text="Freq (1–2000):").pack(side="left")

        # Use Spinbox for constrained numeric entry
        self.freq_var = tk.StringVar(value="100")
        self.freq_spin = tk.Spinbox(
            freqbar,
            from_=1, to=2000, increment=1,
            textvariable=self.freq_var,
            width=7,
            validate="key",
            # Allow only digits
            validatecommand=(self.register(lambda s: s.isdigit() or s == ""), "%P"),
        )
        self.freq_spin.pack(side="left", padx=(6, 6))
        ttk.Button(freqbar, text="Update Freq", command=self._send_update_freq).pack(side="left")

        # Log box
        logframe = ttk.Frame(self, padding=(10, 0, 10, 10))
        logframe.pack(fill="both", expand=True)
        self.log = tk.Text(logframe, wrap="none", height=18, undo=False)
        self.log.configure(state="disabled")
        yscroll = ttk.Scrollbar(logframe, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=yscroll.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        logframe.rowconfigure(0, weight=1)
        logframe.columnconfigure(0, weight=1)

        # Send bar
        sendbar = ttk.Frame(self, padding=(10, 0, 10, 10))
        sendbar.pack(fill="x")
        ttk.Label(sendbar, text="Send:").pack(side="left")
        self.send_var = tk.StringVar()
        self.send_entry = ttk.Entry(sendbar, textvariable=self.send_var)
        self.send_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        self.send_entry.bind("<Return>", lambda e: self._on_send_clicked())
        self.append_nl_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sendbar, text="Append \\n", variable=self.append_nl_var).pack(side="left")
        ttk.Button(sendbar, text="Send", command=self._on_send_clicked).pack(side="left", padx=(8, 0))

        # Nice styles
        style = ttk.Style(self)
        try:
            self.call('tk', 'scaling', 1.0)
        except Exception:
            pass

    def _send_update_freq(self):
        """Send UPDATE_FREQ <value> with bounds checking 1..2000."""
        raw = (self.freq_var.get() or "").strip()
        if not raw.isdigit():
            messagebox.showwarning("Invalid value", "Please enter a number from 1 to 2000.")
            return
        val = int(raw)
        if not (1 <= val <= 2000):
            messagebox.showwarning("Out of range", "Value must be between 1 and 2000.")
            return
        self._send_text(f"UPDATE_FREQ {val}")

    # ---------- Serial handling ----------
    def _refresh_ports(self):
        ports = list_ports.comports()
        items = []
        for p in ports:
            # Display as "COM7 - Bluetooth (ESP32-ELEGOO)" if possible
            desc = f"{p.device} - {p.description}"
            items.append(desc)
        self.port_combo['values'] = items
        if items and not self.port_var.get():
            self.port_var.set(items[0])

    def _get_selected_port(self):
        val = self.port_var.get().strip()
        # Extract the actual device name at the start (e.g., "COM7")
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

    def _on_send_clicked(self):
        text = self.send_var.get().strip()
        if not text:
            return
        self._send_text(text)
        self.send_var.set("")  # clear after send

    def _send_text(self, text):
        if not (self.ser and self.ser.is_open):
            self._log("!! Not connected")
            return
        try:
            payload = text
            if self.append_nl_var.get():
                payload += "\n"
            self.ser.write(payload.encode("utf-8"))
            self._log(f">> {text}")
        except Exception as e:
            self._log(f"!! Send failed: {e}")

    # ---------- RX queue + parsing ----------
    def _poll_rx_queue(self):
        try:
            while True:
                line = self.rx_queue.get_nowait()
                self._log(f"<< {line}")
                self._maybe_update_btn_state(line)
        except queue.Empty:
            pass
        # poll ~60 FPS
        self.after(16, self._poll_rx_queue)

    def _maybe_update_btn_state(self, line: str):
        # Accept formats like "BTN:1" or "HB BTN=1" (from the example firmware)
        s = line.strip().upper()
        val = None
        if s.startswith("BTN:"):
            try:
                val = int(s.split("BTN:")[1].strip()[:1])
            except Exception:
                pass
        elif "HB" in s and "BTN=" in s:
            try:
                part = s.split("BTN=")[1]
                val = int(part.split()[0].strip()[:1])
            except Exception:
                pass

        if val is not None:
            if val == 1:
                self.btn_state_var.set("BTN: PRESSED (1)")
                self.btn_state_label.configure(foreground="#007700")
            else:
                self.btn_state_var.set("BTN: RELEASED (0)")
                self.btn_state_label.configure(foreground="#444444")

    # ---------- Logging ----------
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


