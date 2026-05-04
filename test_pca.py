import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import os
def test_pca(csv_filename):
    df = pd.read_csv(csv_filename)
    df['Time_s'] = (df['Time_ms'] - df['Time_ms'].iloc[0]) / 1000.0
    dt = df['Time_s'].diff().fillna(df['Time_s'].diff().mean()).to_numpy()
    fs = 1.0 / dt.mean()
    print(f"FS: {fs:.1f}")

    ax, ay, az = df['ax'].to_numpy(), df['ay'].to_numpy(), df['az'].to_numpy()
    gx, gy, gz = df['gx'].to_numpy(), df['gy'].to_numpy(), df['gz'].to_numpy()

    # SOS Filter
    sos_resp = butter(4, [0.1, 0.6], btype='band', fs=fs, output='sos')
    def apply_filt(data): return sosfiltfilt(sos_resp, data)

    a_filt = np.column_stack((apply_filt(ax), apply_filt(ay), apply_filt(az)))
    
    # CF
    g_cf = np.zeros((len(df), 3))
    g_init = np.array([ax[0], ay[0], az[0]])
    g_init /= np.linalg.norm(g_init)
    g_cf[0] = g_init
    alpha = 0.98

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

    g_cf_filt = np.column_stack((apply_filt(g_cf[:,0]), apply_filt(g_cf[:,1]), apply_filt(g_cf[:,2])))

    def pca_project(data):
        cov = np.cov(data, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        v = evecs[:, np.argmax(evals)]
        return data.dot(v), (np.max(evals) / np.sum(evals))

    resp_g, var_g = pca_project(a_filt)
    resp_angle_rad, var_a = pca_project(g_cf_filt)
    resp_angle_deg = resp_angle_rad * (180.0 / np.pi)

    print(f"Var G: {var_g:.2f}, Var Angle: {var_a:.2f}")

    # Heart analysis
    sos_heart = butter(4, [0.65, 4.0], btype='band', fs=fs, output='sos')
    h_filt_g = np.column_stack((sosfiltfilt(sos_heart, ax), sosfiltfilt(sos_heart, ay), sosfiltfilt(sos_heart, az)))
    heart_g, var_heart = pca_project(h_filt_g)
    
    # FFT for Heart
    heart_fft = np.abs(np.fft.rfft(heart_g - np.mean(heart_g)))
    heart_freqs = np.fft.rfftfreq(len(df), d=1.0/fs)
    
    dist_resp = int(1.5 * fs)
    peaks_a, _ = find_peaks(resp_angle_deg, distance=dist_resp, prominence=0.0005)
    
    # Displacement on PCA G
    # Integral of acceleration is velocity
    vel = cumulative_trapezoid(resp_g, df['Time_s'].to_numpy(), initial=0)
    # Integral of velocity is displacement
    disp = cumulative_trapezoid(vel, df['Time_s'].to_numpy(), initial=0)
    # Detrend displacement to keep it centered
    from scipy.signal import detrend
    disp = detrend(disp)
    peaks_disp, _ = find_peaks(disp, distance=dist_resp)

    dur = df['Time_s'].iloc[-1] / 60.0
    bpm_a = len(peaks_a)/dur
    bpm_disp = len(peaks_disp)/dur
    print(f"BPM Angle: {bpm_a:.1f}")
    print(f"BPM Disp: {bpm_disp:.1f}")
    
    # Save plot with 10 subplots
    fig, axes = plt.subplots(5, 2, figsize=(15, 18))
    
    t = df['Time_s'].to_numpy()
    
    # Row 1-3: Raw data
    axes[0, 0].plot(t, ax, color='b')
    axes[0, 0].set_title('Accelerometer X (ax)')
    axes[0, 1].plot(t, gx, color='r')
    axes[0, 1].set_title('Gyroscope X (gx)')
    
    axes[1, 0].plot(t, ay, color='b')
    axes[1, 0].set_title('Accelerometer Y (ay)')
    axes[1, 1].plot(t, gy, color='r')
    axes[1, 1].set_title('Gyroscope Y (gy)')
    
    axes[2, 0].plot(t, az, color='b')
    axes[2, 0].set_title('Accelerometer Z (az)')
    axes[2, 1].plot(t, gz, color='r')
    axes[2, 1].set_title('Gyroscope Z (gz)')
    
    # Row 4: Respiratory Analysis
    axes[3, 0].plot(t, disp, color='g')
    if len(peaks_disp) > 0:
        axes[3, 0].plot(t[peaks_disp], disp[peaks_disp], 'x', color='black')
    axes[3, 0].set_title(f'Resp Displacement (PCA G) | {bpm_disp:.1f} BPM')
    
    axes[3, 1].plot(t, resp_angle_deg, color='purple')
    if len(peaks_a) > 0:
        axes[3, 1].plot(t[peaks_a], resp_angle_deg[peaks_a], 'x', color='black')
    axes[3, 1].set_title(f'Resp Angle (CF PCA) | {bpm_a:.1f} BPM')
    
    # Row 5: Cardiac Analysis
    axes[4, 0].plot(t, heart_g, color='crimson')
    axes[4, 0].set_title('Cardiac Signal (PCA G)')
    
    axes[4, 1].plot(heart_freqs, heart_fft, color='orange')
    axes[4, 1].set_xlim(0, 5)
    axes[4, 1].set_title('Cardiac Spectrum (FFT)')
    
    for ax_sub in axes.flat:
        ax_sub.grid(True, alpha=0.3)
    
    plt.suptitle(f"IMU 6-Axis Analysis - {os.path.basename(csv_filename)}", fontsize=16)
    plt.tight_layout()
    output_dir = "plots_imu"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, os.path.basename(csv_filename).replace('.csv', '.png')))
    plt.close()

if __name__ == '__main__':
    import glob
    csv_files = glob.glob("respiratory_*_raw_*.csv")
    for f in csv_files:
        print(f"Processing {f}...")
        try:
            test_pca(f)
        except Exception as e:
            print(f"Failed to process {f}: {e}")

