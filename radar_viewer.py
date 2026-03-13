import serial
import serial.tools.list_ports
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import threading
import numpy as np
from scipy.signal import butter, sosfiltfilt
from collections import deque

class RadarCapture:
    def __init__(self, baud=115200):
        self.baud = baud
        self.serial_port = None
        self.running = False
        self.data_storage = []
        self.live_buffer = deque(maxlen=1000)  # For live plotting
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
                self.live_buffer.append(parts)
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

        # --- SIGNAL PROCESSING (Commented out) ---
        # 1. Time normalization
        df['Time_s'] = (df['Timestamp_ms'] - df['Timestamp_ms'].min()) / 1000.0
        
        # 2. Estimation of Sampling Frequency (fs)
        time_diffs = df['Time_s'].diff().dropna()
        fs = 1.0 / time_diffs.mean() if not time_diffs.empty else 500.0
        print(f"Wyliczona częstotliwość próbkowania: {fs:.2f} Hz")

        """
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
            # Respiratory: 0.05 - 0.5 Hz (3-30 breaths/min)
            sos_resp = get_sos_filter(0.05, 0.5, fs)
            df['Respiratory'] = apply_sos(df['Voltage_mV'], sos_resp)

            # Cardiac: 0.8 - 4.0 Hz (48-240 BPM)
            sos_heart = get_sos_filter(0.8, 4.0, fs)
            df['Cardiac'] = apply_sos(df['Voltage_mV'], sos_heart)
        except Exception as e:
            print(f"Błąd filtrowania: {e}")
            df['Respiratory'] = 0
            df['Cardiac'] = 0

        # 5. FFT Analysis (BPM Calculation)
        N = len(df)
        v_centered = df['Voltage_mV'] - df['Voltage_mV'].mean()
        
        # Respiratory FFT
        resp_fft = np.abs(np.fft.rfft(v_centered))
        resp_freqs = np.fft.rfftfreq(N, d=1.0/fs)
        
        # Band: 0.05 - 0.5 Hz (3 - 30 BPM)
        valid_resp_idx = (resp_freqs >= 0.05) & (resp_freqs <= 0.5)
        if np.any(valid_resp_idx):
            best_resp_freq = resp_freqs[valid_resp_idx][np.argmax(resp_fft[valid_resp_idx])]
            resp_bpm = best_resp_freq * 60.0
        else:
            resp_bpm = 0.0
            best_resp_freq = 0.0

        # Cardiac FFT
        # Use a narrower band for cleaner cardiac peak detection
        valid_heart_idx = (resp_freqs >= 0.8) & (resp_freqs <= 3.0) # 48 - 180 BPM
        if np.any(valid_heart_idx):
            best_heart_freq = resp_freqs[valid_heart_idx][np.argmax(resp_fft[valid_heart_idx])]
            heart_bpm = best_heart_freq * 60.0
        else:
            heart_bpm = 0.0
            best_heart_freq = 0.0
        """

        # --- DEBUG PRINTOUT (Simplified) ---
        print(f"Statystyki sygnału (Voltage_mV):")
        print(f"  Min/Max: {df['Voltage_mV'].min():.2f} / {df['Voltage_mV'].max():.2f} mV")
        print(f"  STD: {df['Voltage_mV'].std():.2f} mV")
        # ----------------------

        # --- PLOTTING (Only Raw Voltage) ---
        plt.figure(figsize=(15, 6))
        plt.plot(df['Time_s'], df['Voltage_mV'], color='blue', linewidth=0.5, label='Raw Signal')
        plt.title(f"HB100 Radar - Raw Voltage Signal ({timestamp_str})")
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [mV]")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
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
            # 15s sleep before starting measurement
            PRE_SLEEP_TIME = 15
            print(f"Waiting {PRE_SLEEP_TIME} seconds before starting measurement...")
            for i in range(PRE_SLEEP_TIME, 0, -1):
                print(f"Starting in: {i}s    ", end="\r", flush=True)
                time.sleep(1)
            
            # Clear data collected during sleep
            app.data_storage.clear()
            app.live_buffer.clear()
            print("\nStarting measurement now!")

            # --- LIVE PLOT SETUP ---
            plt.ion()
            fig_live, ax_live = plt.subplots(figsize=(10, 5))
            line_live, = ax_live.plot([], [], color='blue', linewidth=1)
            ax_live.set_title("Live Voltage Signal")
            ax_live.set_xlabel("Samples")
            ax_live.set_ylabel("Voltage [mV]")
            ax_live.grid(True, alpha=0.3)
            # -----------------------

            print(f"Nagrywanie przez {RECORD_TIME} sekund...")
            start_time = time.time()
            last_plot_time = 0
            
            while time.time() - start_time < RECORD_TIME:
                current_time = time.time()
                elapsed = current_time - start_time
                remaining = int(RECORD_TIME - elapsed)
                
                # Update live plot every 50ms
                if current_time - last_plot_time > 0.05:
                    if app.live_buffer:
                        data = list(app.live_buffer)
                        voltages = [x[2] for x in data]
                        
                        line_live.set_data(range(len(voltages)), voltages)
                        ax_live.set_xlim(0, len(voltages))
                        
                        # Dynamic scaling for Y axis
                        if len(voltages) > 0:
                            v_min, v_max = min(voltages), max(voltages)
                            margin = (v_max - v_min) * 0.1 + 10
                            ax_live.set_ylim(v_min - margin, v_max + margin)
                        
                        plt.pause(0.01)
                    
                    last_plot_time = current_time
                
                print(f"Pozostało: {remaining}s    ", end="\r", flush=True)
                time.sleep(0.01) # Faster loop for smoother UI
            
            print("\nKoniec nagrywania. Wyświetlanie wykresu...")
            plt.close(fig_live)
            plt.ioff()
            app.stop_and_graph()
            
        except KeyboardInterrupt:
            print("\nPrzerwano.")
            plt.close('all')
            app.stop_and_graph()
        except Exception as e:
            print(f"\nBłąd: {e}")
            plt.close('all')
            app.stop_and_graph()
