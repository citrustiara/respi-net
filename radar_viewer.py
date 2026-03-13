import serial
import serial.tools.list_ports
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import threading
import numpy as np
from scipy.signal import butter, sosfiltfilt

class RadarCapture:
    def __init__(self, baud=115200):
        self.baud = baud
        self.serial_port = None
        self.running = False
        self.data_storage = []
        self.read_thread = None

    def list_ports(self):
        return [port.device for port in serial.tools.list_ports.comports()]

    def connect(self):
        ports = self.list_ports()
        if not ports:
            print("Nie znaleziono portów szeregowych.")
            return False
            
        print(f"Dostępne porty: {ports}")
        # Try to find a port (usually the last one or one with ESP32 in name)
        for port in reversed(ports):
            try:
                print(f"Próba połączenia z {port} (Baud: {self.baud})...")
                self.serial_port = serial.Serial(port, self.baud, timeout=1)
                self.running = True
                self.read_thread = threading.Thread(target=self._read_loop)
                self.read_thread.start()
                print("Połączono. Zbieranie danych...")
                return True
            except (serial.SerialException, PermissionError) as e:
                print(f"Błąd połączenia: {e}")
        return False

    def _read_loop(self):
        buffer = ""
        while self.running:
            try:
                if self.serial_port.in_waiting > 0:
                    raw_data = self.serial_port.read(self.serial_port.in_waiting).decode('utf-8', errors='ignore')
                    buffer += raw_data
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            self._process_line(line)
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"Błąd w pętli odczytu: {e}")
                self.running = False

    def _process_line(self, line):
        try:
            # Format: timestamp, raw_adc, voltage_mv
            parts = [float(x) for x in line.split(',')]
            if len(parts) == 3:
                self.data_storage.append(parts)
        except ValueError:
            pass

    def stop_and_graph(self):
        self.running = False
        if self.read_thread:
            self.read_thread.join()
        if self.serial_port:
            self.serial_port.close()

        if len(self.data_storage) < 10:
            print("Za mało danych do wyświetlenia.")
            return

        # Prepare DataFrame and Sort
        df = pd.DataFrame(self.data_storage, columns=['Timestamp_ms', 'RawADC', 'Voltage_mV'])
        df = df.sort_values('Timestamp_ms').reset_index(drop=True)
        
        # Save to CSV
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"radar_raw_{timestamp_str}.csv"
        df.to_csv(filename, index=False)
        print(f"Dane zapisane do {filename}")

        # --- SIGNAL PROCESSING ---
        # 1. Time normalization
        df['Time_s'] = (df['Timestamp_ms'] - df['Timestamp_ms'].min()) / 1000.0
        
        # 2. Estimation of Sampling Frequency (fs)
        time_diffs = df['Time_s'].diff().dropna()
        fs = 1.0 / time_diffs.mean() if not time_diffs.empty else 500.0
        print(f"Wyliczona częstotliwość próbkowania: {fs:.2f} Hz")

        # 3. Butterworth Filter functions
        def get_sos_filter(f_low, f_high, fs_val, order=4):
            # Clip high frequency to Nyquist limit if necessary
            nyq = 0.5 * fs_val
            f_high = min(f_high, nyq * 0.99)
            return butter(order, [f_low, f_high], btype='band', fs=fs_val, output='sos')

        def apply_sos(data, sos):
            return sosfiltfilt(sos, data)

        # 4. Apply Filters
        try:
            # Respiratory: 0.1 - 0.5 Hz (6-30 breaths/min)
            sos_resp = get_sos_filter(0.1, 0.5, fs)
            df['Respiratory'] = apply_sos(df['Voltage_mV'], sos_resp)

            # Cardiac: 0.8 - 4.0 Hz (48-240 BPM)
            sos_heart = get_sos_filter(0.8, 4.0, fs)
            df['Cardiac'] = apply_sos(df['Voltage_mV'], sos_heart)
        except Exception as e:
            print(f"Błąd filtrowania: {e}")
            df['Respiratory'] = 0
            df['Cardiac'] = 0

        # 5. Denoised trend for Top Plot (Moving Average)
        df['Smooth_Trend'] = df['Voltage_mV'].rolling(window=int(fs*0.1), center=True, min_periods=1).mean()

        # --- PLOTTING (3 Subplots) ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # TOP: Raw vs Smooth
        ax1.scatter(df['Time_s'], df['Voltage_mV'], s=1, color='lightskyblue', alpha=0.05, label='Raw Voltage')
        ax1.plot(df['Time_s'], df['Smooth_Trend'], color='blue', linewidth=1, label='Trend (0.1s MA)')
        ax1.set_title(f"HB100 Radar - Signal Decomposition ({timestamp_str})")
        ax1.set_ylabel("Voltage [mV]")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # MIDDLE: Respiratory Signal
        ax2.plot(df['Time_s'], df['Respiratory'], color='green', linewidth=1.5, label='Respiratory (0.1-0.5 Hz)')
        ax2.fill_between(df['Time_s'], df['Respiratory'], color='green', alpha=0.1)
        ax2.set_ylabel("Resp Ampl")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        # BOTTOM: Cardiac Signal
        ax3.plot(df['Time_s'], df['Cardiac'], color='red', linewidth=1, label='Cardiac (0.8-4.0 Hz)')
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Heart Ampl")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

# --- URUCHOMIENIE ---
if __name__ == "__main__":
    RECORD_TIME = 60 # s

    # User requested 921600 baud
    app = RadarCapture(baud=921600)
    
    # Preferred port is COM6
    ports = app.list_ports()
    if "COM6" in ports:
        print("Wymuszanie połączenia z COM6...")
        app.serial_port = serial.Serial("COM6", app.baud, timeout=1)
        app.running = True
        app.read_thread = threading.Thread(target=app._read_loop)
        app.read_thread.start()
        print("Połączono z COM6. Zbieranie danych...")
        success = True
    else:
        success = app.connect()

    if success:
        try:
            print(f"Nagrywanie przez {RECORD_TIME} sekund...")
            start_time = time.time()
            while time.time() - start_time < RECORD_TIME:
                remaining = int(RECORD_TIME - (time.time() - start_time))
                print(f"Pozostało: {remaining}s    ", end="\r", flush=True)
                time.sleep(1)
            
            print("\nKoniec nagrywania. Wyświetlanie wykresu...")
            app.stop_and_graph()
            
        except KeyboardInterrupt:
            print("\nPrzerwano.")
            app.stop_and_graph()
        except Exception as e:
            print(f"\nBłąd: {e}")
            app.stop_and_graph()
