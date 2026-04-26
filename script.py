import serial
import serial.tools.list_ports
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import threading
from scipy.signal import butter, sosfiltfilt
from scipy.integrate import cumulative_trapezoid
import numpy as np

class BreathCapture:
    def __init__(self, baud=115200):
        self.baud = baud
        self.serial_port = None
        self.running = False
        self.data_storage = []
        self.read_thread = None
        self.fs = 0
        self.resp_bpm = 0
        self.heart_bpm = 0

    def list_ports(self):
        return [port.device for port in serial.tools.list_ports.comports()]

    def connect(self, port_name=None):
        ports = self.list_ports()
        if not ports:
            print("Nie znaleziono portów szeregowych.")
            return False
            
        print(f"Dostępne porty: {ports}")
        target_ports = []
        if port_name and port_name in ports:
            target_ports.append(port_name)
        for p in reversed(ports):
            if p not in target_ports:
                target_ports.append(p)

        for port in target_ports:
            try:
                print(f"Próba połączenia z {port} (Baud: {self.baud})...")
                self.serial_port = serial.Serial(port, self.baud, timeout=0.1)
                self.running = True
                self.data_storage = []
                self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
                self.read_thread.start()
                print(f"Połączono z {port}. Zbieranie danych...")
                return True
            except (serial.SerialException, PermissionError) as e:
                print(f"Błąd połączenia z {port}: {e}")
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
                    time.sleep(0.0001)
            except Exception as e:
                print(f"Błąd w pętli odczytu: {e}")
                self.running = False

    def _process_line(self, line):
        try:
            parts = [float(x) for x in line.split(',')]
            if len(parts) == 6:
                # Format: ax, ay, az, gx, gy, gz
                now_ms = time.time() * 1000.0
                self.data_storage.append([now_ms] + parts)
        except ValueError:
            pass

    def stop_and_graph(self):
        self.running = False
        if self.serial_port:
            try:
                self.serial_port.close()
            except:
                pass

        if len(self.data_storage) < 100:
            print("Za mało danych do analizy.")
            return

        print("\nPrzetwarzanie danych...")
        # --- PRZYGOTOWANIE DANYCH ---
        df = pd.DataFrame(self.data_storage, columns=['Time_ms', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"respiratory_6axis_raw_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Dane zapisane do {filename}")
        
        df['Time_s'] = (df['Time_ms'] - df['Time_ms'].iloc[0]) / 1000.0
        dt = df['Time_s'].diff().fillna(df['Time_s'].diff().mean())
        fs = 1.0 / dt.mean()
        print(f"Wykryta częstotliwość próbkowania: {fs:.1f} Hz")

        # --- ANALIZA PCA i FILTROWANIE ---
        ax, ay, az = df['ax'].to_numpy(), df['ay'].to_numpy(), df['az'].to_numpy()
        gx, gy, gz = df['gx'].to_numpy(), df['gy'].to_numpy(), df['gz'].to_numpy()

        def get_sos_filter(f_low, f_high, fs_val, order=4):
            return butter(order, [f_low, f_high], btype='band', fs=fs_val, output='sos')

        def apply_sos(data, sos):
            return sosfiltfilt(sos, data)

        # Pasmo oddechowe: 0.1 - 0.6 Hz
        sos_resp = get_sos_filter(0.1, 0.6, fs)
        a_filt = np.column_stack((apply_sos(ax, sos_resp), apply_sos(ay, sos_resp), apply_sos(az, sos_resp)))

        # Rzutowanie PCA
        def pca_project(data_filt, data_raw):
            cov = np.cov(data_filt, rowvar=False)
            evals, evecs = np.linalg.eigh(cov)
            v = evecs[:, np.argmax(evals)]
            return data_filt.dot(v), data_raw.dot(v)

        a_raw = np.column_stack((ax, ay, az))
        df['Resp_G'], g_raw_proj = pca_project(a_filt, a_raw)
        g_centered = g_raw_proj - g_raw_proj.mean()

        # Pasmo serca: 0.65 - 4.0 Hz
        sos_heart = get_sos_filter(0.65, 4.0, fs)
        h_filt_g = np.column_stack((apply_sos(ax, sos_heart), apply_sos(ay, sos_heart), apply_sos(az, sos_heart)))
        df['Heart_G'], heart_g_raw = pca_project(h_filt_g, a_raw)
        heart_g_centered = heart_g_raw - heart_g_raw.mean()

        # FFT
        N = len(df)
        resp_fft = np.abs(np.fft.rfft(g_centered))
        resp_freqs = np.fft.rfftfreq(N, d=1.0/fs)
        
        valid_resp_idx = (resp_freqs >= 0.1) & (resp_freqs <= 0.6)
        if np.any(valid_resp_idx):
            best_resp_freq = resp_freqs[valid_resp_idx][np.argmax(resp_fft[valid_resp_idx])]
            self.resp_bpm = best_resp_freq * 60.0
        else:
            self.resp_bpm = 0.0
            best_resp_freq = 0.0

        heart_fft = np.abs(np.fft.rfft(heart_g_centered))
        heart_freqs = np.fft.rfftfreq(N, d=1.0/fs)
        valid_heart_idx = (heart_freqs >= 0.65) & (heart_freqs <= 4.0)
        if np.any(valid_heart_idx):
            best_heart_freq = heart_freqs[valid_heart_idx][np.argmax(heart_fft[valid_heart_idx])]
            self.heart_bpm = best_heart_freq * 60.0
        else:
            self.heart_bpm = 0.0
            best_heart_freq = 0.0

        # --- WYKRESY KOŃCOWE ---
        plt.style.use('default')
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f"Analiza IMU: Oddech {self.resp_bpm:.1f} BPM | Tętno {self.heart_bpm:.1f} BPM", fontsize=16)

        t = df['Time_s'].to_numpy()
        axes[0,0].plot(t, ax, label='ax', alpha=0.7)
        axes[0,0].plot(t, ay, label='ay', alpha=0.7)
        axes[0,0].plot(t, az, label='az', alpha=0.7)
        axes[0,0].set_title("Raw Accelerometer")
        axes[0,0].legend()

        axes[0,1].plot(t, gx, label='gx', alpha=0.7)
        axes[0,1].plot(t, gy, label='gy', alpha=0.7)
        axes[0,1].plot(t, gz, label='gz', alpha=0.7)
        axes[0,1].set_title("Raw Gyroscope")
        axes[0,1].legend()

        axes[1,0].plot(t, df['Resp_G'], color='green')
        axes[1,0].set_title("Respiratory Signal (PCA Projected & Filtered)")
        axes[1,0].set_ylabel("G")

        axes[1,1].plot(t, df['Heart_G'], color='crimson')
        axes[1,1].set_title("Cardiac Signal (PCA Projected & Filtered)")
        axes[1,1].set_ylabel("G")

        axes[2,0].plot(resp_freqs, resp_fft)
        axes[2,0].set_xlim(0, 1.0)
        axes[2,0].set_title("Respiratory Spectrum")
        axes[2,0].set_xlabel("Hz")

        axes[2,1].plot(heart_freqs, heart_fft)
        axes[2,1].set_xlim(0, 5.0)
        axes[2,1].set_title("Cardiac Spectrum")
        axes[2,1].set_xlabel("Hz")

        plt.tight_layout()
        plt.show()

def run_breath_capture():
    app = BreathCapture(baud=921600)
    if not app.connect():
        return

    # Live Plot Setup
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.canvas.manager.set_window_title('Real-time IMU Viewer')
    
    # Plot only first 3 axes (accel) for clarity in live view
    lines = [ax.plot([], [], label=l)[0] for l in ['ax', 'ay', 'az']]
    ax.set_title("Live Accelerometer Data")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    is_running = [True]
    def on_stop(event):
        is_running[0] = False
        app.running = False
    
    fig.canvas.mpl_connect('close_event', on_stop)
    fig.canvas.mpl_connect('key_press_event', on_stop)

    print("\n" + "="*50)
    print("Naciśnij DOWOLNY KLAWISZ aby zakończyć pomiar i wyświetlić analizę.")
    print("="*50 + "\n")

    try:
        while is_running[0] and app.running:
            if len(app.data_storage) > 10:
                data = np.array(app.data_storage[-500:]) # Show last 500 samples
                t_rel = np.arange(len(data))
                
                for i in range(3): # ax, ay, az
                    lines[i].set_data(t_rel, data[:, i+1])
                
                ax.set_xlim(0, len(data))
                v_min, v_max = np.min(data[:, 1:4]), np.max(data[:, 1:4])
                margin = (v_max - v_min) * 0.1 + 0.1
                ax.set_ylim(v_min - margin, v_max + margin)
            
            plt.pause(0.01)
            if not plt.fignum_exists(fig.number):
                break
    except KeyboardInterrupt:
        print("\nPrzerwano.")
    finally:
        app.running = False
        plt.close(fig)
        app.stop_and_graph()

if __name__ == "__main__":
    run_breath_capture()