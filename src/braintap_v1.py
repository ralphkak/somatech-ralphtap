# pip install pyserial
import serial, time

# Change this to your Bluetooth COM port on Windows (e.g., 'COM7')
# On macOS/Linux, something like '/dev/tty.ESP32-ELEGOO-SerialPort' or '/dev/rfcomm0'
PORT = 'COM12'

with serial.Serial(PORT, 115200, timeout=1) as ser:
    time.sleep(1)  # give it a moment
    ser.write(b'LED ON\n')
    print(ser.readline().decode(errors='ignore').strip())
    ser.write(b'READ\n')
    print(ser.readline().decode(errors='ignore').strip())
    time.sleep(1.2)
    print(ser.readline().decode(errors='ignore').strip())
    ser.write(b'LED OFF\n')
    print(ser.readline().decode(errors='ignore').strip())
