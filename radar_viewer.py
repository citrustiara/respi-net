import serial
import serial.tools.list_ports
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import threading
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from collections import deque

class RadarCapture:
    def __init__(self, baud=115200):
        self.baud = baud
        self.serial_port = None
        self.running = False
        self.data_storage = []
        self.live_buffer = deque(maxlen=2048)  # Larger buffer for better FFT resolution
        self.read_thread = None
        self.start_time_real = None

    def list_ports(self):
        return [port.device for port in serial.tools.list_ports.comports()]

    def connect(self, port_name=None):
        ports = self.list_ports()
        if not ports:
            print("Nie znaleziono portów szeregowych.")
            return False
            
        print(f"Dostępne porty: {ports}")
        
        # Priority: requested port, then others in reverse order
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
                self.live_buffer.clear()
                self.start_time_real = time.time()
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
                    raw_bytes = self.serial_port.read(self.serial_port.in_waiting)
                    raw_data = raw_bytes.decode('utf-8', errors='ignore')
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
            # Expected format: timestamp_ms, raw_adc, voltage_mv
            parts = [float(x) for x in line.split(',')]
            if len(parts) == 3:
                self.data_storage.append(parts)
                self.live_buffer.append(parts)
        except ValueError:
            pass

    def stop(self):
        self.running = False
        if self.serial_port:
            try:
                self.serial_port.close()
            except:
                pass
        print("\nZatrzymano zbieranie danych.")

    def save_and_plot_final(self):
        if len(self.data_storage) < 10:
            print("Za mało danych do wyświetlenia.")
            return

        # Prepare DataFrame
        df = pd.DataFrame(self.data_storage, columns=['Timestamp_ms', 'RawADC', 'Voltage_mV'])
        df = df.sort_values('Timestamp_ms').reset_index(drop=True)
        
        # Save to CSV
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"radar_raw_{timestamp_str}.csv"
        df.to_csv(filename, index=False)
        print(f"Dane zapisane do {filename}")

        # Signal Analysis
        df['Time_s'] = (df['Timestamp_ms'] - df['Timestamp_ms'].min()) / 1000.0
        time_diffs = df['Time_s'].diff().dropna()
        fs = 1.0 / time_diffs.mean() if not time_diffs.empty else 500.0
        print(f"Średnia częstotliwość próbkowania: {fs:.2f} Hz")

        v = df['Voltage_mV'].values
        v_detrended = v - np.mean(v)
        
        # Final FFT using Welch for better spectral estimation
        nperseg = min(len(v), 4096)
        freqs, psd = welch(v_detrended, fs, nperseg=nperseg)

        # Visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        ax1.plot(df['Time_s'], df['Voltage_mV'], color='#1f77b4', linewidth=0.5)
        ax1.set_title(f"HB100 Radar - Time Domain Signal ({timestamp_str})", fontsize=14)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Voltage [mV]")
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(freqs, psd, color='#ff7f0e', linewidth=1)
        ax2.set_title("HB100 Radar - Power Spectral Density (Welch)", fontsize=14)
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("Power/Frequency [V^2/Hz]")
        ax2.grid(True, alpha=0.3, which='both')
        
        # Doppler Speed Secondary Axis (HB100: 10.525 GHz)
        # fd = 2 * v * (ft / c) => v = fd * c / (2 * ft)
        # For 10.525GHz: v (m/s) = fd / 70.16
        ax2_speed = ax2.twiny()
        ax2_speed.set_xlim(ax2.get_xlim())
        xticks = ax2.get_xticks()
        ax2_speed.set_xticks(xticks)
        ax2_speed.set_xticklabels([f"{x/70.16:.1f}" for x in xticks])
        ax2_speed.set_xlabel("Estimated Speed [m/s] (Doppler shift)")

        plt.tight_layout()
        print("Wyświetlanie końcowego wykresu...")
        plt.show()

def run_radar_viewer():
    # User requested high baud rate for ESP32 ADC streaming
    app = RadarCapture(baud=921600)
    
    # Try COM6 first as requested by user previously, then fallback
    if not app.connect(port_name="COM6"):
        print("Nie udało się połączyć z COM6, szukam innych portów...")
        if not app.connect():
            print("Błąd: Nie można połączyć z żadnym portem.")
            return

    # Modern Dark UI
    plt.style.use('dark_background')
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(12, 8))
    fig.canvas.manager.set_window_title('Radar Viewer')
    
    line_time, = ax_time.plot([], [], color='#00ffcc', linewidth=1, label='Raw ADC')
    ax_time.set_title("Live Radar Signal (Voltage)", fontsize=12)
    ax_time.set_ylabel("mV")
    ax_time.grid(True, color='#333333', alpha=0.5)
    ax_time.legend(loc='upper right')

    line_freq, = ax_freq.plot([], [], color='#ff3399', linewidth=1.5, label='Spectrum')
    ax_freq.set_title("Live Frequency Spectrum (FFT)", fontsize=12)
    ax_freq.set_xlabel("Frequency [Hz]")
    ax_freq.set_ylabel("Magnitude")
    ax_freq.grid(True, color='#333333', alpha=0.5)
    ax_freq.legend(loc='upper right')
    
    # Text info on plot
    info_text = ax_time.text(0.02, 0.95, '', transform=ax_time.transAxes, verticalalignment='top', color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

    is_running = [True]

    def on_close(event):
        is_running[0] = False
        app.running = False

    def on_key(event):
        # Stop on any key press as requested
        print(f"\nZatrzymywanie (klawisz: {event.key})...")
        is_running[0] = False
        app.running = False

    fig.canvas.mpl_connect('close_event', on_close)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print("\n" + "!"*60)
    print("  RADAR VIEWER URUCHOMIONY")
    print("  - Zamykanie okna lub naciśnięcie DOWOLNEGO KLAWISZA kończy pomiar")
    print("  - Po zakończeniu zostanie wygenerowany pełny raport i wykres")
    print("!"*60 + "\n")

    last_fs = 500.0
    
    try:
        while is_running[0] and app.running:
            if len(app.live_buffer) > 64:
                data = list(app.live_buffer)
                voltages = np.array([x[2] for x in data])
                timestamps = np.array([x[0] for x in data])
                
                # Dynamic sampling rate estimation
                if len(timestamps) > 1:
                    dt_avg = np.mean(np.diff(timestamps)) / 1000.0
                    if dt_avg > 0:
                        last_fs = 1.0 / dt_avg

                # Update Time Domain Plot
                display_len = min(len(voltages), 1000)
                v_disp = voltages[-display_len:]
                line_time.set_data(range(display_len), v_disp)
                ax_time.set_xlim(0, display_len)
                v_min, v_max = np.min(v_disp), np.max(v_disp)
                margin = max((v_max - v_min) * 0.1, 50)
                ax_time.set_ylim(v_min - margin, v_max + margin)

                # Update Frequency Domain Plot (FFT)
                # Use a power-of-two window for efficiency
                fft_size = 1024
                if len(voltages) >= fft_size:
                    v_fft = voltages[-fft_size:]
                else:
                    v_fft = voltages
                
                # Hanning window to avoid spectral leakage
                win = np.hanning(len(v_fft))
                v_win = (v_fft - np.mean(v_fft)) * win
                
                fft_vals = np.abs(np.fft.rfft(v_win)) / (len(v_fft)/2)
                fft_freqs = np.fft.rfftfreq(len(v_fft), d=1.0/last_fs)
                
                line_freq.set_data(fft_freqs, fft_vals)
                ax_freq.set_xlim(0, min(last_fs / 2, 250)) # Zoom to 250Hz max for typical movement
                ax_freq.set_ylim(0, np.max(fft_vals[1:]) * 1.2 + 0.1)
                
                # Detect peak frequency (ignore DC)
                if len(fft_vals) > 5:
                    peak_idx = np.argmax(fft_vals[5:]) + 5 
                    peak_freq = fft_freqs[peak_idx]
                    peak_mag = fft_vals[peak_idx]
                    
                    if peak_mag > 5: # Threshold for "significant" movement
                        speed = peak_freq / 70.16
                        info_text.set_text(f"Status: RUCH WYKRYTY\nCzęstotliwość: {peak_freq:.1f} Hz\nPrędkość ok.: {speed:.2f} m/s")
                    else:
                        info_text.set_text("Status: BRAK WYRAŹNEGO RUCHU")

            plt.pause(0.01)
            
            # Check if figure is still open
            if not plt.fignum_exists(fig.number):
                is_running[0] = False

    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika (Ctrl+C).")
    finally:
        app.stop()
        plt.close('all')
        app.save_and_plot_final()

if __name__ == "__main__":
    run_radar_viewer()
