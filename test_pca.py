import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt

def test_pca():
    df = pd.read_csv("respiratory_6axis_raw_2026-03-08_03-01-12.csv")
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

    from scipy.signal import find_peaks
    from scipy.integrate import cumulative_trapezoid

    dist_resp = int(1.5 * fs)
    peaks_a, _ = find_peaks(resp_angle_deg, distance=dist_resp, prominence=0.0005)
    
    # Try displacement on PCA G
    disp = cumulative_trapezoid(resp_g, df['Time_s'].to_numpy(), initial=0)
    peaks_disp, _ = find_peaks(disp, distance=dist_resp, prominence=0.0005)

    dur = df['Time_s'].iloc[-1] / 60.0
    print(f"BPM Angle: {len(peaks_a)/dur:.1f}")
    print(f"BPM Disp: {len(peaks_disp)/dur:.1f}")

if __name__ == '__main__':
    test_pca()
