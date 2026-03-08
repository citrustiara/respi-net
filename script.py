import serial
import serial.tools.list_ports
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import threading
from scipy.signal import find_peaks, butter, sosfiltfilt
from scipy.integrate import cumulative_trapezoid
import numpy as np

class BreathCapture:
    def __init__(self, baud=115200):
        self.baud = baud
        self.serial_port = None
        self.running = False
        self.data_storage = []
        self.read_thread = None
        # Wyniki analizy (dla testów)
        self.fs = 0
        self.resp_bpm = 0
        self.heart_bpm = 0

    def list_ports(self):
        return [port.device for port in serial.tools.list_ports.comports()]

    def connect(self):
        ports = self.list_ports()
        if not ports:
            print("Nie znaleziono portów szeregowych.")
            return False
            
        print(f"Dostępne porty: {ports}")
        for port in ports:
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
            parts = [float(x) for x in line.split(',')]
            if len(parts) == 6:
                # Otrzymujemy raw [ax, ay, az, gx, gy, gz]
                now_ms = time.time() * 1000.0
                self.data_storage.append([now_ms] + parts)
        except ValueError:
            pass

    def stop_and_graph(self):
        self.running = False
        if self.read_thread:
            self.read_thread.join()
        if self.serial_port:
            self.serial_port.close()

        if len(self.data_storage) < 100:
            print("Za mało danych do analizy.")
            return

        # --- PRZYGOTOWANIE DANYCH ---
        df = pd.DataFrame(self.data_storage, columns=['Time_ms', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df.to_csv(f"respiratory_6axis_raw_{timestamp}.csv", index=False)
        
        df['Time_s'] = (df['Time_ms'] - df['Time_ms'].iloc[0]) / 1000.0
        dt = df['Time_s'].diff().fillna(df['Time_s'].diff().mean())
        fs = 1.0 / dt.mean()
        print(f"\nWykryta częstotliwość próbkowania: {fs:.1f} Hz")

        # --- ANALIZA PCA (OPTYMALNA, NIEZALEŻNA OD ORIENTACJI) ---
        ax, ay, az = df['ax'].to_numpy(), df['ay'].to_numpy(), df['az'].to_numpy()
        gx, gy, gz = df['gx'].to_numpy(), df['gy'].to_numpy(), df['gz'].to_numpy()

        # 1. Filtry SOS
        def get_sos_filter(f_low, f_high, fs_val, order=4):
            return butter(order, [f_low, f_high], btype='band', fs=fs_val, output='sos')

        def apply_sos(data, sos):
            win = max(3, int(fs / 50))
            smoothed = pd.Series(data).rolling(window=win, center=True, min_periods=1).mean().fillna(0).to_numpy()
            return sosfiltfilt(sos, smoothed)

        sos_resp = get_sos_filter(0.1, 0.6, fs)
        
        # Filtrujemy przyspieszenie
        a_filt = np.column_stack((apply_sos(ax, sos_resp), apply_sos(ay, sos_resp), apply_sos(az, sos_resp)))

        # 2. Filtr komplementarny 3D 
        alpha = 0.98 
        g_cf = np.zeros((len(df), 3))
        g_init = np.array([ax[0], ay[0], az[0]])
        g_init /= np.linalg.norm(g_init)
        g_cf[0] = g_init
        
        for i in range(1, len(df)):
            w = np.array([np.radians(gx[i]), np.radians(gy[i]), np.radians(gz[i])])
            g_old = g_cf[i-1]
            cx = w[1]*g_old[2] - w[2]*g_old[1]
            cy = w[2]*g_old[0] - w[0]*g_old[2]
            cz = w[0]*g_old[1] - w[1]*g_old[0]
            g_pred = g_old - np.array([cx, cy, cz]) * dt[i]
            
            norm_pred = np.linalg.norm(g_pred)
            if norm_pred > 0: g_pred /= norm_pred
            
            a_meas = np.array([ax[i], ay[i], az[i]])
            a_norm = np.linalg.norm(a_meas)
            if a_norm > 0: a_meas /= a_norm
            else: a_meas = g_pred
            
            g_cf_i = alpha * g_pred + (1 - alpha) * a_meas
            g_cf_i /= np.linalg.norm(g_cf_i)
            g_cf[i] = g_cf_i

        g_cf_filt = np.column_stack((apply_sos(g_cf[:,0], sos_resp), apply_sos(g_cf[:,1], sos_resp), apply_sos(g_cf[:,2], sos_resp)))

        # 3. Rzutowanie PCA (znalezienie optymalnej osi ruchu)
        def pca_project(data_filt, data_raw):
            cov = np.cov(data_filt, rowvar=False)
            evals, evecs = np.linalg.eigh(cov)
            v = evecs[:, np.argmax(evals)]  # Główna oś
            return data_filt.dot(v), data_raw.dot(v)

        a_raw = np.column_stack((ax, ay, az))
        df['Resp_G'], g_raw_proj = pca_project(a_filt, a_raw)
        
        df['Resp_Angle'], angle_raw_proj = pca_project(g_cf_filt, g_cf)
        df['Resp_Angle'] *= (180.0 / np.pi)
        angle_raw_proj *= (180.0 / np.pi)

        # Centrowanie raw pod wykres
        g_centered = g_raw_proj - g_raw_proj.mean()
        angle_centered = angle_raw_proj - angle_raw_proj.mean()

        # 4. Tętno (Serce) - PCA w paśmie kardiologicznym
        sos_heart = get_sos_filter(0.75, 3.0, fs)
        
        # Filtrowanie przyspieszenia (G) w paśmie serca
        h_filt_g = np.column_stack((apply_sos(ax, sos_heart), apply_sos(ay, sos_heart), apply_sos(az, sos_heart)))
        df['Heart_G'], _ = pca_project(h_filt_g, a_raw)
        
        # Filtrowanie kąta (Tilt) w paśmie serca
        h_filt_angle = np.column_stack((apply_sos(g_cf[:,0], sos_heart), apply_sos(g_cf[:,1], sos_heart), apply_sos(g_cf[:,2], sos_heart)))
        df['Heart_Angle'], _ = pca_project(h_filt_angle, g_cf)
        df['Heart_Angle'] *= (180.0 / np.pi)

        # --- PRZEMIESZCZENIE ---
        t = df['Time_s'].to_numpy()
        df['Resp_Disp'] = cumulative_trapezoid(df['Resp_G'], t, initial=0)
        df['Heart_Disp'] = cumulative_trapezoid(df['Heart_G'], t, initial=0)

        # --- DETEKCJA PEAKÓW ---
        # ODDECH: Teraz brany z PRZEMIESZCZENIA (Resp_Disp)
        dist_resp = int(1.5 * fs)
        resp_peaks, _ = find_peaks(df['Resp_Disp'], distance=dist_resp, prominence=0.0005)
        
        # TĘTNO: Wykrywanie peaków (wyłączone do dalszych poprawek)
        # dist_heart = int(0.4 * fs)
        # initial_heart_peaks, _ = find_peaks(df['Heart_G'], distance=dist_heart, prominence=0.003)
        # ... logic commented out ...
        heart_peaks = []

        duration_min = df['Time_s'].iloc[-1] / 60.0
        self.fs = fs
        self.resp_bpm = len(resp_peaks) / duration_min if duration_min > 0 else 0
        self.heart_bpm = 0 # len(heart_peaks) / duration_min if duration_min > 0 else 0

        # print(f"Analiza zakończona. Oddech: {self.resp_bpm:.1f} BPM, Tętno: {self.heart_bpm:.1f} BPM.")
        print(f"Analiza zakończona. Oddech: {self.resp_bpm:.1f} BPM.")

        # --- WYKRESY ---
        fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
        # fig.suptitle(f"Analiza oddechu: {self.resp_bpm:.1f} BPM | Tętno: {self.heart_bpm:.1f} BPM", fontsize=16, fontweight='bold')
        fig.suptitle(f"Analiza oddechu: {self.resp_bpm:.1f} BPM | Sygnał serca", fontsize=16, fontweight='bold')

        # KOLUMNA 1: ODDECH
        axes[0,0].plot(t, angle_centered, color='blue', alpha=0.15, label='Kąt 3D (Raw)')
        axes[0,0].plot(t, df['Resp_Angle'], color='blue', linewidth=2, label='Filtr 0.1-0.6Hz')
        axes[0,0].set_title("Oddech: Zmiana Orientacji (°) - Wypadkowa", fontsize=12)
        axes[0,0].set_ylabel("Stopnie")
        y_min, y_max = df['Resp_Angle'].min(), df['Resp_Angle'].max()
        margin = (y_max - y_min) * 0.3 if y_max != y_min else 1.0
        axes[0,0].set_ylim(y_min - margin, y_max + margin)
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()

        axes[1,0].plot(t, g_centered, color='green', alpha=0.15, label='G Wypadkowa (Raw)')
        axes[1,0].plot(t, df['Resp_G'], color='green', linewidth=2, label='Filtr 0.1-0.6Hz')
        axes[1,0].set_title("Oddech: Zgęstki G (Wypadkowa 3-osiowa)", fontsize=12)
        axes[1,0].set_ylabel("G")
        y_min, y_max = df['Resp_G'].min(), df['Resp_G'].max()
        margin = (y_max - y_min) * 0.3 if y_max != y_min else 0.05
        axes[1,0].set_ylim(y_min - margin, y_max + margin)
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()

        axes[2,0].plot(t, df['Resp_Disp'], color='green', linewidth=2)
        axes[2,0].plot(t[resp_peaks], df['Resp_Disp'].iloc[resp_peaks], "ro", label='Peak Oddechu')
        axes[2,0].fill_between(t, df['Resp_Disp'], alpha=0.2, color='green')
        axes[2,0].set_title("Oddech: Przemieszczenie (∫G) - Detekcja BPM", fontsize=12)
        axes[2,0].set_xlabel("Czas [s]")
        axes[2,0].grid(True, alpha=0.3)
        axes[2,0].legend()

        # KOLUMNA 2: TĘTNO
        axes[0,1].plot(t, df['Heart_Angle'], color='crimson', linewidth=1.5, label='Filtr 0.8-5Hz')
        axes[0,1].set_title("Serce: Zmiana Orientacji (°) - Wypadkowa", fontsize=12)
        axes[0,1].set_ylabel("Stopnie")
        y_min, y_max = df['Heart_Angle'].min(), df['Heart_Angle'].max()
        margin = (y_max - y_min) * 0.2 if y_max != y_min else 0.005
        axes[0,1].set_ylim(y_min - margin, y_max + margin)
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()

        axes[1,1].plot(t, df['Heart_G'], color='crimson', linewidth=1.5, alpha=0.8)
        # axes[1,1].plot(t[heart_peaks], df['Heart_G'].iloc[heart_peaks], "kx", markersize=6, label='Skurcz')
        axes[1,1].set_title("Serce: Zgęstki G (Wypadkowa 3-osiowa)", fontsize=12)
        axes[1,1].set_ylabel("G")
        y_min, y_max = df['Heart_G'].min(), df['Heart_G'].max()
        margin = (y_max - y_min) * 0.2 if y_max != y_min else 0.005
        axes[1,1].set_ylim(y_min - margin, y_max + margin)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()

        axes[2,1].plot(t, df['Heart_Disp'], color='purple', linewidth=1.5)
        axes[2,1].fill_between(t, df['Heart_Disp'], alpha=0.2, color='purple')
        axes[2,1].set_title("Tętno: Mikroruchy (∫G_cardio)", fontsize=12)
        axes[2,1].set_ylabel("j.u.")
        axes[2,1].set_xlabel("Czas [s]")
        axes[2,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# --- URUCHOMIENIE ---
if __name__ == "__main__":
    WAIT_TIME = 15
    RECORD_TIME = 45

    app = BreathCapture(baud=921600)
    if app.connect():
        try:
            print(f"\n--- STABILIZACJA: Oczekiwanie {WAIT_TIME} sekund przed startem... ---")
            for i in range(WAIT_TIME, 0, -1):
                print(f"Start za: {i}s   ", end="\r", flush=True)
                time.sleep(1)
            
            app.data_storage = []
            
            print(f"\n--- START POMIARU: Nagrywanie przez {RECORD_TIME} sekund... ---")
            for i in range(RECORD_TIME, 0, -1):
                print(f"Pozostało: {i}s    ", end="\r", flush=True)
                time.sleep(1)
            
            print("\n--- KONIEC: Przetwarzanie danych... ---")
            app.stop_and_graph()
            
        except KeyboardInterrupt:
            print("\nPrzerwano ręcznie.")
            plt.close('all')
            app.running = False
            if app.serial_port:
                app.serial_port.close()
        except Exception as e:
            print(f"\nWystąpił błąd: {e}")
            app.stop_and_graph()