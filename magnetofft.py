import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import h5py
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
import polars as pl
import pandas as pd
from pathlib import Path
from matplotlib.colors import LogNorm
from matplotlib import colors,cm
import os

#################################################
# Reading environmental logs
#################################################

def load_monitorpi_csv(filepath):
    df = pd.read_csv(filepath)
    # Convert 'localtime' from string to datetime
    df["timestamp"] = pd.to_datetime(df["localtime"], format="%Y_%m_%d_%H_%M")
    return df

def plot_monitorpi_data(df):
    time = df["timestamp"]
    columns_to_plot = [
        ("extvolt(V)", "External Voltage (V)"),
        ("cputemp(C)", "CPU Temperature (°C)"),
        ("envtemperature(C)", "Environment Temperature (°C)"),
        ("envhumidity(%)", "Environment Humidity (%)"),
        ("envpressure(hPa)", "Environment Pressure (hPa)")
    ]

    num_plots = len(columns_to_plot)
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 2.5 * num_plots), sharex=True)
    
    for ax, (col, label) in zip(axs, columns_to_plot):
        ax.plot(time, df[col], marker="o", linestyle="-")
        ax.set_ylabel(label)
        ax.grid(True)

    axs[-1].set_xlabel("Time")
    fig.suptitle("Raspberry Pi Sensor Readings Over Time", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.xticks(rotation=45)
    plt.show()

def plot_monitorpi_fromcsv(filepath):
    df = load_monitorpi_csv(filepath)
    plot_monitorpi_data(df)

#################################################
# Loading data
#################################################
plt.rcParams.update({'font.size': 14})

SLOPE=1.0083421207668117
OFFSET=-295.30861087311496

def calibrate_data(dataraw):
    data = np.zeros_like(dataraw,dtype=float)
    if dataraw.dtype == np.uint16:
        # calibrate only int16 data
        data = (dataraw * SLOPE) + OFFSET
        # convert into voltage
        data = ( data / 65536.0 ) * 20.0 -10.0
    else:
        data = dataraw
        print("No calibration, assuming data is calibrated.")
    return data
    
def read_hdf5_times(pathh5,get_startime=False):
    """
    Read 'end_time' and 'measure_time' from the 'voltage' dataset attributes
    in an HDF5 file without loading the dataset itself.
    
    Returns
        (end_time, measure_time)
    """
    with h5py.File(pathh5, 'r') as f:
        attrs = f['voltage'].attrs
        end_time = pd.to_datetime(attrs['end_time'], format="%Y_%m_%d_%H_%M")
        measure_time = attrs['measure_time']
        if get_startime:
            start_time = pd.to_datetime(attrs['start_time'], format="%Y_%m_%d_%H_%M")
            return end_time, measure_time, start_time
        else:
            return end_time, measure_time
    
def load_hdf5(pathh5):
    '''
    load hdf5 and extract metadata, convert into magnetic field value in μT
    '''
    f = h5py.File(pathh5,'r')
    data = f['voltage']
    dset = {}
    
    dset['sample_rate'] = data.attrs['sample_rate']
    dset['measure_time'] = data.attrs['measure_time']
    dset['end_time'] = data.attrs['end_time']
    dset['start_time'] = data.attrs['start_time']
    
    data = np.array(data)
    data = calibrate_data(data)
    data = data * 1000./143
    

    if data.ndim == 1 or data.shape[1] == 1:
        # Single-channel case
        dset['x'] = data.flatten()
    else:
        # Multi-channel case, assuming channel order 0, 1, 4 -> x, z, y
        dset['x'] = data[:, 0]
        dset['y'] = data[:, 2]
        dset['z'] = data[:, 1]   

    return dset

def load_csv_pl(pathcsv):
    df = pl.read_csv(pathcsv)
    data = df.to_numpy()
    data = calibrate_data(data)
    data = data * 1000. / 143.

    dset = {}

    if data.ndim == 1 or data.shape[1] == 1:
        # Single-channel case
        dset['x'] = data.flatten()
    else:
        # Multi-channel case, assuming channel order 0, 1, 4 -> x, z, y
        dset['x'] = data[:, 0]
        dset['y'] = data[:, 2]
        dset['z'] = data[:, 1]

    return dset
    
def load_csv(pathcsv):
    '''
    load csv magnetometer result and convert into magnetic field value in μT
    '''
    data = np.genfromtxt(pathcsv,delimiter=',')
    data = calibrate_data(data)
    data = data * 1000. / 143.
    dset = {}
    if data.ndim == 1 or data.shape[1] == 1:
        # Single-channel case
        dset['x'] = data.flatten()
    else:
        # Multi-channel case, assuming channel order 0, 1, 4 -> x, z, y
        dset['x'] = data[:, 0]
        dset['y'] = data[:, 2]
        dset['z'] = data[:, 1]
    
    return dset

#################################################
# FFT Power Spectrum prelim (not used!)
#################################################

def fft_timeseries(chs,samplerate):
    n = len(chs)
    freq = np.fft.fftfreq(n, d=1/samplerate)
    fft_values = np.fft.fft(chs)
    return freq, fft_values.real, fft_values.imag

def fft_power_spectrum(chs,samplerate):
    freq, real, imag = fft_timeseries(chs, samplerate)
    power_spectrum = real**2 + imag**2
    
    dt = 1. / samplerate
    power_spectrum = power_spectrum * dt**2 / 2. /np.pi
    return freq, power_spectrum

def fft_LSD(chs,samplerate):
    freq, real, imag = fft_timeseries(chs, samplerate)
    power_spectrum = real**2 + imag**2
    
    dt = 1. / samplerate
    N = len(chs)
    psd = power_spectrum * dt / N / 2. / np.pi  # Convert to power spectral density
    LSD = np.sqrt(psd)
    return freq, LSD

def fft_amplitude(chs, samplerate):
    freq, real, imag = fft_timeseries(chs, samplerate)
    amplitude = np.sqrt(real**2 + imag**2)
    N = len(chs)
    amplitude = amplitude / N * 2  # Normalize amplitude
    return freq, amplitude

def slide_window_average(data,window_array):
    # use np.convolve
    return convolve(data, window_array, mode='wrap')

def gaussian_stft(chs, samplerate, g_std = 1000, g_length = 10000, mfft = 1000, hop=1000):
    fs = samplerate
    N = len(chs)
    t_x = np.arange(N) * 1.0/ fs
    w = gaussian(g_length,std=g_std,sym=True)
    SFT = ShortTimeFFT(w, hop=hop, fs=fs, mfft=mfft, scale_to='magnitude')
    Sx = SFT.stft(chs)

    extent = SFT.extent(N)
    window_width = SFT.m_num*SFT.T
    sigma_t = g_std*SFT.T
    delta_t = SFT.delta_t
    delta_f = SFT.delta_f

    res = {}
    res['sx'] = Sx
    res['extent'] = extent
    res['window_width'] = window_width
    res['sigma_t'] = sigma_t
    res['delta_t'] = delta_t
    res['delta_f'] = delta_f

    return res

#################################################
# Signal Preprocessing
#################################################

# def lowpass(chs,fs,fmax):

#################################################
# FFT Power Spectrum and Power Spectral Density
#################################################

def compute_psd(chs,fs,nodc=True):
    # remove DC
    chs = chs - np.mean(chs)
    N = len(chs)
    
    # add window
    ws = np.hamming(N)
    ws1 = np.sum(ws)
    ws2 = np.sum(ws**2)

    # rfft
    f = np.fft.rfftfreq(N,1./fs)
    ak = np.fft.rfft(chs*ws)

    # normalize into psd
    psd = np.abs(ak)**2/fs/ws2
    psd[1:-1] *= 2
    if nodc:
        f = f[1:]
        psd = psd[1:]
    return f,psd

def compute_ps(chs,fs,nodc=True):
    # remove DC
    chs = chs - np.mean(chs)
    N = len(chs)
    
    # add window
    ws = np.hamming(N)
    ws1 = np.sum(ws)
    ws2 = np.sum(ws**2)

    # rfft
    f = np.fft.rfftfreq(N,1./fs)
    ak = np.fft.rfft(chs*ws)

    # normalize into psd
    ps = np.abs(ak)**2/ws1**2
    ps[1:-1] *= 2
    if nodc:
        f = f[1:]
        ps = ps[1:]
    return f,ps

def compute_ensemble_psd(chs,fs,Lbin:int,overlapratio:float=0.5,nodc=True,fmax:float=None):
    Nhop = max(1, int(round(Lbin * (1.0 - overlapratio))))
    N = len(chs)
    nbins = (N-Lbin)//Nhop
    
    f = np.fft.rfftfreq(Lbin,1./fs)
    nf = len(f)

    ensemb = np.zeros((nbins,nf))

    for n in range(nbins):
        binmin = n*Nhop
        binmax = binmin+Lbin
        _,psd = compute_psd(chs[binmin:binmax],fs,nodc=False)
        ensemb[n] = psd

    # avg_psd = np.mean(ensemb,axis=0)
    if nodc:
        f = f[1:]
        ensemb = ensemb[:,1:]

    if fmax is not None:
        mask = f < fmax
        ensemb = ensemb[mask]
        f = f[mask]

    return f,ensemb
        
def compute_averaged_psd(chs,fs,Lbin:int,overlapratio:float=0.5,nodc=True,fmax:float=None):
    Nhop = max(1, int(round(Lbin * (1.0 - overlapratio))))
    N = len(chs)
    nbins = (N-Lbin)//Nhop
    
    f = np.fft.rfftfreq(Lbin,1./fs)
    nf = len(f)

    ensemb = np.zeros((nbins,nf))

    for n in range(nbins):
        binmin = n*Nhop
        binmax = binmin+Lbin
        _,psd = compute_psd(chs[binmin:binmax],fs,nodc=False)
        ensemb[n] = psd

    avg_psd = np.mean(ensemb,axis=0)
    if nodc:
        f = f[1:]
        avg_psd = avg_psd[1:]

    if fmax is not None:
        mask = f < fmax
        avg_psd = avg_psd[mask]
        f = f[mask]
        
    return f,avg_psd

#################################################
# psd from-file save to file
#################################################

def segment_indices(N: int, Lbin: int, overlap: float) -> np.ndarray:
    """
    Return start indices for overlapping segments of length Lbin.
    overlap: fraction in [0,1). overlap=0.5 -> 50% overlap.
    """
    if not (0 <= overlap < 1):
        raise ValueError("overlap must be in [0,1).")
    if Lbin <= 0 or Lbin > N:
        raise ValueError("Lbin must be >0 and <= N.")
    hop = max(1, int(round(Lbin * (1.0 - overlap))))
    # Number of complete segments that fit
    nbins = 1 + (N - Lbin) // hop
    starts = np.arange(nbins) * hop
    return starts.astype(int)

def ps_psd_from_windowed_fft(x_win: np.ndarray, fs: float, window: np.ndarray,
                             nodc: bool = True,fmax:float=None):
    """
    Given a *windowed* real segment x_win (length L), compute:
      - rFFT frequencies
      - one-sided PS (|FFT|^2 normalization)
      - one-sided PSD (per Hz)
    Normalizations:
        PS  = |A_k|^2 / (sum(window))^2
        PSD = |A_k|^2 / (fs * sum(window^2))
    Both are doubled for bins 1..-2 (one-sided). DC & Nyquist kept then removed if nodc.
    """
    L = len(x_win)
    if len(window) != L:
        raise ValueError("window length must equal segment length.")

    # rFFT
    A = np.fft.rfft(x_win)
    f = np.fft.rfftfreq(L, d=1.0/fs)

    ws1 = np.sum(window)
    ws2 = np.sum(window**2)

    # Raw two-sided power
    pow_spec = np.abs(A)**2

    # PS and PSD (two-sided)
    ps  = pow_spec / (ws1**2)
    psd = pow_spec / (fs * ws2)

    # Convert to one-sided (double interior bins for real signals)
    if len(ps) > 2:
        ps[1:-1]  *= 2.0
        psd[1:-1] *= 2.0

    if nodc:
        f   = f[1:]
        ps  = ps[1:]
        psd = psd[1:]

    if fmax is not None:
        mask = f < fmax
        f = f[mask]
        ps = ps[mask]
        psd = psd[mask]

    return f, ps, psd

# -------------------------------------
# Main pipeline: load -> segment -> save
# -------------------------------------

def save_windowed_ps_psd_hdf5(
    h5_in: str,
    out_dir: str | None = None,
    channels=('x','y','z'),
    Lbin: int = None,
    overlap: float = 0.5,
    nodc: bool = True,
    fmax: float = None,
    fmin:float = None
):
    """
    1) Load HDF5 via your load_hdf5()
    2) Segment into overlapping Hamming windows (length Lbin, fraction 'overlap')
    3) Compute PS and PSD per segment for requested channels
    4) Save structured HDF5 with:
        /spectral (attrs: sample_rate, Lbin, overlap, window, nodc, start_time, end_time, measure_time)
        /spectral/{ch}/
            ps           [nbins, nf]
            psd          [nbins, nf]
            frequency    [nf]
            seg_start_ix [nbins]
            seg_start_unix [nbins]   (UNIX seconds for segment start time)
    """
        
    # ---- Load raw data & metadata
    dset = load_hdf5(h5_in)  # uses your function (returns dict)
    fs = float(dset['sample_rate'])
    
    if Lbin is None:
        if fmin is not None:
            # print('! hint, no Lbin, taken from fmin')
            N = len(dset['x'])
            Lbin = int(round(fs / fmin))
        else:
            print('! Warning, no Lbin, taken default')
            N = len(dset['x'])
            Lbin = N//20
    
    end_time_str = dset['end_time']     # e.g. "YYYY_MM_DD_HH_MM"
    measure_time = float(dset['measure_time'])  # seconds

    # times
    end_ts = pd.to_datetime(end_time_str, format="%Y_%m_%d_%H_%M")
    start_ts = end_ts - pd.to_timedelta(measure_time, unit='s')
    start_unix = start_ts.value / 1e9  # convert ns -> s

    # Choose output path
    in_path = Path(h5_in)
    out_dir_path = Path(out_dir) if out_dir is not None else in_path.parent
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # infer filename from input; ensure .hdf5 extension
    if in_path.suffix.lower() in (".h5", ".hdf5"):
        out_name = f"{in_path.stem}_spectral.hdf5"
    else:
        # input isn't .h5/.hdf5 — keep name and add suffix
        out_name = f"{in_path.name}_spectral.hdf5"

    h5_out = out_dir_path / out_name

    # Validate presence of channels
    available = [k for k in dset.keys() if k in ('x','y','z')]
    chans = [ch for ch in channels if ch in available]
    if not chans:
        raise ValueError(f"No requested channels found in input. Available: {available}")

    # Prepare segmentation on the *longest* requested channel
    N = len(dset[chans[0]])
    if any(len(dset[ch]) != N for ch in chans):
        raise ValueError("All requested channels must have the same length.")

    starts = segment_indices(N, Lbin=Lbin, overlap=overlap)
    nbins = len(starts)
    window = np.hamming(Lbin)

    # Frequency vector from a dummy segment (we’ll store the one after nodc handling)
    # build once
    dummy_segment = (dset[chans[0]][starts[0]:starts[0]+Lbin] - np.mean(dset[chans[0]][starts[0]:starts[0]+Lbin]))
    f_dummy, _, _ = ps_psd_from_windowed_fft(dummy_segment * window, fs, window, nodc=nodc,fmax=fmax)
    nf = len(f_dummy)

    # Segment absolute times
    seg_start_unix = start_unix + starts / fs

    # ---- Write out results
    with h5py.File(h5_out, "w") as fout:
        grp = fout.create_group("spectral")
        # Global attrs
        grp.attrs['sample_rate'] = fs
        grp.attrs['Lbin'] = int(Lbin)
        grp.attrs['overlap'] = float(overlap)
        grp.attrs['window'] = "hamming"
        grp.attrs['nodc'] = bool(nodc)
        grp.attrs['measure_time'] = float(measure_time)
        grp.attrs['start_time'] = start_ts.strftime("%Y_%m_%d_%H_%M")
        grp.attrs['end_time'] = end_time_str

        # Create a shared frequency dataset at top-level for convenience
        grp.create_dataset("frequency", data=f_dummy)

        # Also store segment indexing & times
        grp.create_dataset("seg_start_ix", data=starts, dtype='i8')
        grp.create_dataset("seg_start_unix", data=seg_start_unix, dtype='f8')

        # Per-channel groups
        for ch in chans:
            ch_grp = grp.create_group(ch)

            # Allocate matrices
            ps_mat  = np.empty((nbins, nf), dtype=np.float64)
            psd_mat = np.empty((nbins, nf), dtype=np.float64)

            x = np.asarray(dset[ch], dtype=float)

            for i, s0 in enumerate(starts):
                seg = x[s0:s0+Lbin]
                # detrend / remove DC *before* windowing (as in your funcs)
                seg = seg - np.mean(seg)
                xw = seg * window
                f, ps, psd = ps_psd_from_windowed_fft(xw, fs, window, nodc=nodc,fmax=fmax)
                # f is same for all, already stored as /spectral/frequency
                ps_mat[i, :]  = ps
                psd_mat[i, :] = psd

            ch_grp.create_dataset("ps", data=ps_mat)
            ch_grp.create_dataset("psd", data=psd_mat)

    print(f"[ok] Saved windowed PS/PSD to: {h5_out}")

def save_averaged_ps_psd_hdf5(
    h5_in: str,
    out_dir: str | None = None,
    channels=('x','y','z'),
    Lbin: int = None,
    overlap: float = 0.5,
    nodc: bool = True,
    fmax: float = None,
    fmin:float = None
):

    dset = load_hdf5(h5_in)
    fs = float(dset['sample_rate'])
    
    if Lbin is None:
        if fmin is not None:
            N = len(dset['x'])
            Lbin = int(round(fs / fmin))
        else:
            # print('! Warning, no Lbin, taken default')
            N = len(dset['x'])
            Lbin = N//20
    
    end_time_str = dset['end_time']
    start_time_str = dset['start_time']
    measure_time = dset['measure_time']

    # times
    end_ts = pd.to_datetime(end_time_str, format="%Y_%m_%d_%H_%M")
    start_ts = pd.to_datetime(start_time_str, format="%Y_%m_%d_%H_%M")
    start_unix = start_ts.value / 1e9
    
    # Choose output path
    in_path = Path(h5_in)
    out_dir_path = Path(out_dir) if out_dir is not None else in_path.parent
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # infer filename from input; ensure .hdf5 extension
    if in_path.suffix.lower() in (".h5", ".hdf5"):
        out_name = f"{in_path.stem}_spectral.hdf5"
    else:
        # input isn't .h5/.hdf5 — keep name and add suffix
        out_name = f"{in_path.name}_spectral.hdf5"

    h5_out = out_dir_path / out_name

    # Validate presence of channels
    available = [k for k in dset.keys() if k in ('x','y','z')]
    chans = [ch for ch in channels if ch in available]
    if not chans:
        raise ValueError(f"No requested channels found in input. Available: {available}")

    # Prepare segmentation on the *longest* requested channel
    N = len(dset[chans[0]])
    if any(len(dset[ch]) != N for ch in chans):
        raise ValueError("All requested channels must have the same length.")

    starts = segment_indices(N, Lbin=Lbin, overlap=overlap)
    nbins = len(starts)
    window = np.hamming(Lbin)

    # Frequency vector from a dummy segment (we’ll store the one after nodc handling)
    # build once
    dummy_segment = (dset[chans[0]][starts[0]:starts[0]+Lbin] - np.mean(dset[chans[0]][starts[0]:starts[0]+Lbin]))
    f_dummy, _, _ = ps_psd_from_windowed_fft(dummy_segment * window, fs, window, nodc=nodc,fmax=fmax)
    nf = len(f_dummy)

    # Segment absolute times
    seg_start_unix = start_unix + starts / fs

    # ---- Write out results
    with h5py.File(h5_out, "w") as fout:
        grp = fout.create_group("spectral")
        # Global attrs
        grp.attrs['sample_rate'] = fs
        grp.attrs['Lbin'] = int(Lbin)
        grp.attrs['overlap'] = float(overlap)
        grp.attrs['window'] = "hamming"
        grp.attrs['nodc'] = bool(nodc)
        grp.attrs['measure_time'] = float(measure_time)
        grp.attrs['start_time'] = start_time_str
        grp.attrs['end_time'] = end_time_str

        # Create a shared frequency dataset at top-level for convenience
        grp.create_dataset("frequency", data=f_dummy)

        # Also store segment indexing & times
        grp.create_dataset("seg_start_ix", data=starts, dtype='i8')
        grp.create_dataset("seg_start_unix", data=seg_start_unix, dtype='f8')

        # Per-channel groups
        for ch in chans:
            ch_grp = grp.create_group(ch)

            # Allocate matrices
            ps_mat  = np.empty((nbins, nf), dtype=np.float64)
            psd_mat = np.empty((nbins, nf), dtype=np.float64)

            x = np.asarray(dset[ch], dtype=float)

            for i, s0 in enumerate(starts):
                seg = x[s0:s0+Lbin]
                # detrend / remove DC *before* windowing (as in your funcs)
                seg = seg - np.mean(seg)
                xw = seg * window
                f, ps, psd = ps_psd_from_windowed_fft(xw, fs, window, nodc=nodc,fmax=fmax)
                # f is same for all, already stored as /spectral/frequency
                ps_mat[i, :]  = ps
                psd_mat[i, :] = psd

            ps_avg = np.mean(ps_mat,axis=0)
            psd_avg = np.mean(psd_mat,axis=0)
            ch_grp.create_dataset("ps", data=ps_avg)
            ch_grp.create_dataset("psd", data=psd_avg)

    print(f"[ok] Saved averaged PS/PSD to: {h5_out}")

def load_averaged_ps_psd(h5_path: str, channel: str = "x"):

    with h5py.File(h5_path, "r") as f:
        if "spectral" not in f:
            raise ValueError("File does not contain /spectral group. Is this the right HDF5?")

        g = f["spectral"]
        if channel not in ("x","y","z","amp","avg"):
            raise ValueError(f"Channel '{channel}' not found.")

        # read global attrs
        fs = float(g.attrs["sample_rate"])
        Lbin = int(g.attrs["Lbin"])
        overlap = float(g.attrs["overlap"])
        nodc = bool(g.attrs["nodc"])
        start_time_str = g.attrs.get("start_time", None)
        end_time_str   = g.attrs.get("end_time", None)
        measure_time   = float(g.attrs.get("measure_time", np.nan))

        # frequency axis and segment timing
        frequency = g["frequency"][:]


        # matrices
        if channel in ("x","y","z"):
            ps  = g[channel]["ps"][:]
            psd = g[channel]["psd"][:]
        elif channel =="amp":
            ps_x = g["x"]["ps"][:]
            ps_y = g["y"]["ps"][:]
            ps_z = g["z"]["ps"][:]
            psd_x = g["x"]["psd"][:]
            psd_y = g["y"]["psd"][:]
            psd_z = g["z"]["psd"][:]
            ps = ps_x+ps_y+ps_z
            psd = psd_x+psd_y+psd_z
        else:
            # output avg of three directions
            ps_x = g["x"]["ps"][:]
            ps_y = g["y"]["ps"][:]
            ps_z = g["z"]["ps"][:]
            psd_x = g["x"]["psd"][:]
            psd_y = g["y"]["psd"][:]
            psd_z = g["z"]["psd"][:]
            ps = (ps_x+ps_y+ps_z)/3.0
            psd = (psd_x+psd_y+psd_z)/3.0

        meta = {
            "sample_rate": fs,
            "Lbin": Lbin,
            "overlap": overlap,
            "nodc": nodc,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "measure_time": measure_time,
            "n_freq": ps.shape[0]
        }

    return ps, psd, frequency, meta

def load_windowed_ps_psd(h5_path: str, channel: str = "x"):
    """
    Load per-segment PS and PSD matrices plus segment-center datetimes
    from the HDF5 produced by `save_windowed_ps_psd_hdf5`.

    Parameters
    ----------
    h5_path : str
        Path to the spectral HDF5 file.
    channel : {"x","y","z"}
        Which channel group to read.

    Returns
    -------
    ps : np.ndarray, shape (n_segments, n_freq)
        One-sided power spectrum (windowed-segment PS).
    psd : np.ndarray, shape (n_segments, n_freq)
        One-sided power spectral density (per Hz).
    seg_center_times : pd.DatetimeIndex, length n_segments
        Center time of each segment as timezone-naive pandas timestamps.
    frequency : np.ndarray, shape (n_freq,)
        One-sided frequency axis corresponding to the matrix columns.
        (If `nodc=True` during save, DC is omitted.)
    meta : dict
        Useful metadata: sample_rate, Lbin, overlap, nodc, start_time, end_time, measure_time.
    """
    with h5py.File(h5_path, "r") as f:
        if "spectral" not in f:
            raise ValueError("File does not contain /spectral group. Is this the right HDF5?")

        g = f["spectral"]
        if channel not in ("x","y","z","amp","avg"):
            raise ValueError(f"Channel '{channel}' not found.")

        # read global attrs
        fs = float(g.attrs["sample_rate"])
        Lbin = int(g.attrs["Lbin"])
        overlap = float(g.attrs["overlap"])
        nodc = bool(g.attrs["nodc"])
        start_time_str = g.attrs.get("start_time", None)
        end_time_str   = g.attrs.get("end_time", None)
        measure_time   = float(g.attrs.get("measure_time", np.nan))

        # frequency axis and segment timing
        frequency = g["frequency"][:]
        seg_start_ix = g["seg_start_ix"][:]
        seg_start_unix = g["seg_start_unix"][:]

        # center time = start + (Lbin/2)/fs
        center_unix = seg_start_unix + (Lbin / (2.0 * fs))
        seg_center_times = pd.to_datetime(center_unix, unit="s")

        # matrices
        if channel in ("x","y","z"):
            ps  = g[channel]["ps"][:]
            psd = g[channel]["psd"][:]
        elif channel == "amp":
            ps_x = g["x"]["ps"][:]
            ps_y = g["y"]["ps"][:]
            ps_z = g["z"]["ps"][:]
            psd_x = g["x"]["psd"][:]
            psd_y = g["y"]["psd"][:]
            psd_z = g["z"]["psd"][:]
            ps = ps_x+ps_y+ps_z
            psd = psd_x+psd_y+psd_z
        else:
            ps_x = g["x"]["ps"][:]
            ps_y = g["y"]["ps"][:]
            ps_z = g["z"]["ps"][:]
            psd_x = g["x"]["psd"][:]
            psd_y = g["y"]["psd"][:]
            psd_z = g["z"]["psd"][:]
            ps = (ps_x+ps_y+ps_z)/3.0
            psd = (psd_x+psd_y+psd_z)/3.0

        meta = {
            "sample_rate": fs,
            "Lbin": Lbin,
            "overlap": overlap,
            "nodc": nodc,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "measure_time": measure_time,
            "n_segments": ps.shape[0],
            "n_freq": ps.shape[1],
        }

    return ps, psd, seg_center_times, frequency, meta
    
#################################################
# PLotting
#################################################

def plot_sample_ps(path,fs=1000.,ax=None,label=None,orientation=['x','y','z'],alpha=0.6):
    if path.endswith('.hdf5'):
        print('->loading hdf5...',end='\r')
        dset = load_hdf5(path)
    if path.endswith('.csv'):
        print('->loading csv ...',end='\r')
        dset = load_csv_pl(path)
    print('->load complete   ',end='\r')

    if ax is None:
        fig,ax = plt.subplots(figsize = (8,6))
        ax.set_xlabel('f[Hz]')
        ax.set_ylabel(r'FFT Amplitude[$\mu T$]')
    else:
        fig=None
        
    for vec in orientation:
        print('->plotting direction '+vec,end='\r')
        data = dset[vec]
        f,ps = compute_ps(data,fs)
        ax.loglog(f,np.sqrt(ps),label=label+vec,alpha=alpha)
        
    print('plot complete '+path[-20:-4]+'.')
    return fig,ax

def plot_sample_psd(path,fs=1000.,ax=None,label=None,orientation=['x','y','z'],Lbin=None,overlap=0.5,alpha=0.6,fmax:float=None,fmin:float=None):
    print('plot sample '+path+':')
    
    if path.endswith('.hdf5'):
        print('->loading hdf5...',end='\r')
        dset = load_hdf5(path)
    if path.endswith('.csv'):
        print('->loading csv ...',end='\r')
        dset = load_csv_pl(path)
    print('->load complete   ',end='\r')

    if Lbin is None:
        if fmin is not None:
            print('! hint, no Lbin, taken from fmin')
            N = len(dset['x'])
            Lbin = int(round(fs / fmin))
        else:
            print('! Warning, no Lbin, taken default')
            N = len(dset['x'])
            Lbin = N//20
    
    if ax is None:
        fig,ax = plt.subplots(figsize = (8,6))
        ax.set_xlabel('f[Hz]')
        ax.set_ylabel(r'LSD[$\mu T/ \sqrt{Hz}$]')
    else:
        fig=None
    
    for vec in orientation:
        print('->plotting direction '+vec,end='\r')
        data = dset[vec]
        f,ps = compute_averaged_psd(data,fs,Lbin,overlap,fmax=fmax)
        ax.loglog(f,np.sqrt(ps),label=vec,alpha=alpha)
        
    print('plot complete '+path+'.')
    return fig,ax

#################################################
# Spectral batch analysis
#################################################

def spectral_average(
    h5_path: str,
    channel: str = "x",
    which: str = "psd",            # {"ps", "psd"}
):

    ps, psd, seg_center_times, f, meta = load_windowed_ps_psd(h5_path, channel=channel)
        
    ps = np.mean(ps,axis=0)
    psd = np.mean(psd,axis=0)
    ps = np.sqrt(ps)
    psd = np.sqrt(psd)
    if which =='psd':
        return psd,meta
    elif which =='ps':
        return ps,meta

def spectral_load_average(
    h5_path: str,
    channel: str = 'x',
    which:str = 'psd'
):
    ps, psd, frequency, meta = load_averaged_ps_psd(h5_path,channel=channel)

    ps = np.sqrt(ps)
    psd = np.sqrt(psd)
    if which =='psd':
        return psd,meta
    elif which =='ps':
        return ps,meta
        

def spectral_directory_read(
    h5_dir:str|None = None,
    h5_pathlist = None,
    channel: str = "x",
    which:str = 'psd',
    min_timespan: float|None = None,
    time_of_day: tuple = None,
    averaged_sample: bool=False
):
    if h5_dir is not None:
        names = sorted(os.listdir(h5_dir))
        h5_pathlist = [
            os.path.join(h5_dir, n)
            for n in names
            if n.lower().endswith((".h5", ".hdf5"))
        ]
    elif h5_pathlist is None:
        raise ValueError("Must provide either a data directory (h5_dir) or a list of paths (h5_pathlist).")

    spectrum_list = []
    time_center_list = []
    for h5_path in h5_pathlist:
        try:
            if averaged_sample == False:
                spectrum, meta = spectral_average(h5_path, channel=channel, which=which)
            else:
                spectrum, meta = spectral_load_average(h5_path,channel=channel,which=which)
        except Exception as e:
            print(f"! skip {h5_path}: {e}")
            continue
            
        fmt = "%Y_%m_%d_%H_%M"
        start_time = pd.to_datetime(meta["start_time"], format=fmt)
        end_time = pd.to_datetime(meta["end_time"],format=fmt)
        centre_time = start_time + (end_time - start_time)/2
        hour = centre_time.hour
        measure_time = (end_time - start_time).seconds
        
        if (min_timespan is not None) and (measure_time < float(min_timespan)):
            print(f'! warn: {h5_path} too short (measure_time={measure_time}s < {min_timespan}s); skipping')
            continue
        

        
        if time_of_day is not None:
            lo, hi = time_of_day
            if not (0 <= lo <= 23 and 0 <= hi <= 23):
                raise ValueError("time_of_day must be a pair of ints in [0, 23].")

            # Inclusive range; handle overnight windows (e.g., 22→6)
            if lo <= hi:
                in_window = (lo <= hour <= hi)
            else:
                in_window = (hour >= lo) or (hour <= hi)

            if not in_window:
                continue
                
        spectrum_list.append(spectrum)
        time_center_list.append(centre_time)
                
        if len(time_center_list) > 1:
            order = np.argsort(np.array(time_center_list, dtype="datetime64[ns]"))
            spectrum_list = [spectrum_list[i] for i in order]
            time_center_list = [time_center_list[i] for i in order]
        
    return spectrum_list,time_center_list

def _xedges_from_centers(f: np.ndarray,strictly_positive=True) -> np.ndarray:
    """Build pcolormesh x-edges from center frequencies (linear scale)."""
    f = np.asarray(f, float)
    df = np.diff(f)
    # interior edges: midpoints
    mids = f[:-1] + 0.5 * df
    # extrapolate first/last edge
    first = f[0] - 0.5 * df[0] if len(df) else f[0] - 0.5
    last  = f[-1] + 0.5 * df[-1] if len(df) else f[0] + 0.5
    if not strictly_positive:
        return np.concatenate([[first], mids, [last]])
    else:
        return np.concatenate([mids, [last]])

def compute_amplitude_hist_per_frequency(
    spectra_list: list[np.ndarray],
    f: np.ndarray,
    val_bins: int | np.ndarray = 200,
    val_range: tuple[float, float] | None = None,
    log_val: bool = True,
    density: bool = False,
):
    # ---- Consistency check
    f = np.asarray(f, float)
    Nf = f.size
    for i, S in enumerate(spectra_list):
        if np.asarray(S).ndim != 1 or len(S) != Nf:
            raise ValueError(f"spectra_list[{i}] has length {len(S)} != Nf={Nf}")

    # ---- Stack to (M, Nf)
    S = np.vstack([np.asarray(s, float) for s in spectra_list])  # M x Nf

    # ---- Choose amplitude range
    if val_range is None:
        # robust percentiles across all positive values (useful if log bins)
        flat = S.ravel()
        if log_val:
            flat = flat[np.isfinite(flat) & (flat > 0)]
            vmin, vmax = (np.percentile(flat, 1.0), np.percentile(flat, 99.5)) if flat.size else (1e-12, 1.0)
            vmin = max(vmin, np.finfo(float).tiny)
        else:
            flat = flat[np.isfinite(flat)]
            vmin, vmax = (np.percentile(flat, 1.0), np.percentile(flat, 99.5)) if flat.size else (0.0, 1.0)
        if vmax <= vmin:
            vmax = vmin * (10.0 if log_val else 1.01)
    else:
        vmin, vmax = val_range

    # ---- Build amplitude bin edges
    if isinstance(val_bins, int):
        if log_val:
            yedges = np.logspace(np.log10(vmin), np.log10(vmax), val_bins + 1)
        else:
            yedges = np.linspace(vmin, vmax, val_bins + 1)
    else:
        yedges = np.asarray(val_bins, float)

    Ny = len(yedges) - 1
    H = np.zeros((Nf, Ny), dtype=float)

    # ---- Per-frequency histograms (only along amplitude)
    for j in range(Nf):
        col = S[:, j]
        m = np.isfinite(col)
        if log_val:
            m &= (col > 0)
        col = col[m]
        if col.size:
            hj, _ = np.histogram(col, bins=yedges, density=density)
            H[j, :] = hj  # store for frequency j
        # else row remains zeros

    return H, yedges

def plot_amplitude_hist_heatmap(
    H: np.ndarray,
    f: np.ndarray,
    yedges: np.ndarray,
    log_counts: bool = True,
    which: str = "psd",   # {"ps","psd"} for axis label units
    ax=None,
    strictly_positive=True
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    xedges = _xedges_from_centers(f,strictly_positive=strictly_positive)
    # H is (Nf, Ny); pcolormesh expects (Ny, Nf) when using (xedges,yedges)
    Z = H.T

    if strictly_positive:
        Z = Z[:,1:]

    norm = LogNorm() if (log_counts and np.any(Z > 0)) else None
    mesh = ax.pcolormesh(xedges, yedges, Z, shading="auto", norm=norm, rasterized=True)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Counts" if norm is None else "Counts")

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("LSD [$\\mu\\mathrm{T}/\\sqrt{\\mathrm{Hz}}$]" if which.lower()=="psd"
                  else "Amplitude [$\\mu\\mathrm{T}$]")
    ax.set_yscale("log")  # amplitude axis commonly spans decades
    ax.set_xscale("log")
    # ax.set_title("Per-frequency amplitude histograms")
    return fig, ax, mesh

def plot_percentile(
    spectra_list: list[np.ndarray],
    f: np.ndarray,
    time_center_list: list,
    *,
    color_mid: str = 'k',
    color_fill: str = 'k',
    color_bound: str = 'k',
    use_db: bool = False,
    title:str|None = None,
    ylim: list[float] = [1e-5,2e-1],
    label:str|None = None,
    ax=None,
    strictly_positive=True
):
    if len(spectra_list) != len(time_center_list):
        raise ValueError("spectra_list and time_center_list must have the same length.")
    if not isinstance(f, np.ndarray) or f.ndim != 1:
        raise ValueError("f must be a 1D numpy array.")
    nfreq = f.size
    for i, s in enumerate(spectra_list):
        if not isinstance(s, np.ndarray) or s.size != nfreq:
            raise ValueError(f"spectra_list[{i}] has length {s.size}, expected {nfreq}.")

    spectra_array = np.array(spectra_list)
    spectral_16,spectral_50,spectral_84 = np.percentile(spectra_array,[16,50,84],axis=0)

    if strictly_positive:
        f = f[1:]
        spectral_16 = spectral_16[1:]
        spectral_50 = spectral_50[1:]
        spectral_84 = spectral_84[1:]
    
    if use_db:
        spectral_16 = 20.0 * np.log10(np.clip(spectral_16, tiny, None))
        spectral_50 = 20.0 * np.log10(np.clip(spectral_50, tiny, None))
        spectral_84 = 20.0 * np.log10(np.clip(spectral_84, tiny, None))
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.fill_between(f,spectral_16,spectral_84,color=color_fill,alpha=0.1)
    ax.plot(f,spectral_16,color = color_bound,alpha=0.2,linewidth=0.4)
    ax.plot(f,spectral_84,color = color_bound,alpha=0.2,linewidth=0.4)
    ax.plot(f,spectral_50,color = color_mid,alpha=0.8,label=label,linewidth=1.2)
    ax.set_ylim(ylim[0],ylim[1])
    ax.set_title(title)
    ax.set_xscale("log")
    
    if use_db:
        ax.set_ylabel("LSD [dB]")
    else:
        ax.set_ylabel("LSD [$\\mu\\mathrm{T}/\\sqrt{\\mathrm{Hz}}$]")
        ax.set_yscale("log")

    ax.set_xlabel('f [Hz]')
    return fig,ax

def plot_daynight_cycle(
    spectra_list: list[np.ndarray],
    f: np.ndarray,
    time_center_list: list,
    *,
    use_db: bool = False,             # plot PSD in dB if True; otherwise linear (log y-scale)
    db_floor: float | None = None,    # e.g., -200 (dB). If None, no explicit floor beyond tiny clip
    cmap: str = "twilight",           # cyclic colormap: 'twilight', 'twilight_shifted', or 'hsv'
    alpha: float = 0.2,               # line transparency
    lw: float = 0.8,                  # line width
    time_zone: str | None = None,     # e.g., "Europe/Zurich"; converts tz-aware times or localizes naive
    fmin: float | None = None,
    fmax: float | None = None,
    sort_by_hour: bool = False,       # draw order by hour (optional aesthetic)
    title: str | None = None,
    ax=None,
):
    if len(spectra_list) != len(time_center_list):
        raise ValueError("spectra_list and time_center_list must have the same length.")
    if not isinstance(f, np.ndarray) or f.ndim != 1:
        raise ValueError("f must be a 1D numpy array.")
    nfreq = f.size
    for i, s in enumerate(spectra_list):
        if not isinstance(s, np.ndarray) or s.size != nfreq:
            raise ValueError(f"spectra_list[{i}] has length {s.size}, expected {nfreq}.")

    # Convert times to pandas Series for easy hour extraction and timezone handling
    tser = pd.to_datetime(pd.Series(time_center_list), errors="coerce")
    if tser.isna().any():
        bad = np.where(tser.isna())[0][:5]
        raise ValueError(f"Found invalid timestamps at indices {bad.tolist()} (showing up to 5).")

    if time_zone is not None:
        if tser.dt.tz is None:
            tser = tser.dt.tz_localize(time_zone)
        else:
            tser = tser.dt.tz_convert(time_zone)

    # Hour as float in [0, 24)
    hour_float = (
        tser.dt.hour
        + tser.dt.minute / 60.0
        + tser.dt.second / 3600.0
        + tser.dt.microsecond / 3.6e9
    ).astype(float)

    # Optional draw order
    order = np.argsort(hour_float.values) if sort_by_hour else np.arange(len(spectra_list))

    # Map hour -> color (0..24)
    norm = colors.Normalize(vmin=0.0, vmax=24.0, clip=True)
    cmap_obj = cm.get_cmap(cmap)

    # Prepare figure/axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    # Plot each spectrum
    tiny = np.finfo(float).tiny
    for idx in order:
        col = cmap_obj(norm(hour_float.iloc[idx]))
        y = spectra_list[idx]

        if use_db:
            y_db = 20.0 * np.log10(np.clip(y, tiny, None))
            if db_floor is not None:
                y_db = np.maximum(y_db, db_floor)
            ax.plot(f, y_db, alpha=alpha, lw=lw, color=col)
        else:
            ax.plot(f, y, alpha=alpha, lw=lw, color=col)

    # Axes formatting
    ax.set_xlabel("Frequency [Hz]")
    if use_db:
        ax.set_ylabel("LSD [dB]")
    else:
        ax.set_ylabel("LSD [$\\mu\\mathrm{T}/\\sqrt{\\mathrm{Hz}}$]")
        ax.set_yscale("log")

    if fmin is not None or fmax is not None:
        ax.set_xlim(left=fmin if fmin is not None else ax.get_xlim()[0],
                    right=fmax if fmax is not None else ax.get_xlim()[1])

    ax.set_xscale("log")  # typical for PSD vs frequency
    ax.grid(True, which="both", ls=":", alpha=0.3)

    if title:
        ax.set_title(title)

    # Colorbar keyed to local time
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Local time (hour)")
    cbar.set_ticks([0, 6, 12, 18, 24])
    cbar.set_ticklabels(["00", "06", "12", "18", "24"])

    return fig, ax

def plot_2d_spectral_histogram(
    h5_path: str,
    channel: str = "x",
    which: str = "psd",            # {"ps", "psd"}
    freq_bins: int | np.ndarray = 200,
    val_bins: int | np.ndarray = 200,
    freq_range: tuple[float,float] | None = None,
    val_range: tuple[float,float] | None = None,
    log_freq: bool = True,
    log_val: bool = True,          # log-scale y (PS/PSD) using log-spaced bins
    log_counts: bool = True,       # log color scale for the bincounts
    density: bool = False,         # pass to np.histogram2d
    ax=None,
):
    ps, psd, seg_center_times, f, meta = load_windowed_ps_psd(h5_path, channel=channel)
    ps = np.sqrt(ps)
    psd = np.sqrt(psd)
    Z = psd if which.lower() == "psd" else ps
    if which.lower() not in ("ps", "psd"):
        raise ValueError("which must be 'ps' or 'psd'")

    # 2) Flatten to point cloud (freq, value)
    #    Repeat f for each row of Z and ravel both
    F = np.broadcast_to(f, Z.shape)
    x = F.ravel()
    y = Z.ravel()

    # 3) Clean / mask and choose ranges
    m = np.isfinite(x) & np.isfinite(y)
    if log_val:
        m &= (y > 0)  # log requires strictly positive
    x = x[m]
    y = y[m]

    # Frequency range
    if freq_range is None:
        if log_freq:
            x_pos = x[x > 0]
            xmin, xmax = float(np.min(x_pos)), float(np.max(x_pos)) 
        else:
            xmin, xmax = float(np.min(x)), float(np.max(x))
    else:
        xmin, xmax = freq_range

    # Value (PS/PSD) range with robust clipping
    if val_range is None:
        # robust percentiles
        if log_val:
            y_pos = y[y > 0]
            ymin, ymax = np.percentile(y_pos, [1.0, 99.5]) if y_pos.size else (1e-12, 1.0)
        else:
            ymin, ymax = np.percentile(y, [1.0, 99.5]) if y.size else (0.0, 1.0)
        # avoid degenerate edges
        if ymax <= ymin:
            ymax = ymin * (10.0 if log_val else 1.01)
    else:
        ymin, ymax = val_range

    # 4) Build bin edges
    
    if isinstance(freq_bins, int):
        if log_freq:
            xmin = max(xmin, np.finfo(float).tiny)
            xedges = np.logspace(np.log10(xmin), np.log10(xmax),
                                 freq_bins + 1)
        else:
            xedges = np.linspace(xmin, xmax, freq_bins + 1)
    else:
        xedges = np.asarray(freq_bins, dtype=float)

    if isinstance(val_bins, int):
        if log_val:
            ymin = max(ymin, np.finfo(float).tiny)
            yedges = np.logspace(np.log10(ymin), np.log10(ymax), val_bins + 1)
        else:
            yedges = np.linspace(ymin, ymax, val_bins + 1)
    else:
        yedges = np.asarray(val_bins, dtype=float)

    # 5) Histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=[xedges, yedges], density=density)

    # 6) Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    norm = LogNorm() if log_counts and np.any(H > 0) else None
    mesh = ax.pcolormesh(xedges, yedges, H.T, shading="auto", norm=norm)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Counts" if not density else "Density")

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("LSD [$\\mu\\mathrm{T}/\\sqrt{\\mathrm{Hz}}$]" if which.lower()=="psd" else "rt(PS) [$\\mu\\mathrm{T}$]")

    if log_val:
        ax.set_yscale("log")

    if log_freq:
        ax.set_xscale("log")

    # Title with basics
    stem = Path(h5_path).stem
    ax.set_title(f"{stem} · ch={channel} · {which.upper()} · "
                 f"Lbin={meta.get('Lbin','?')} · overlap={meta.get('overlap','?')}")

    ax.grid(False)
    fig.tight_layout()
    return fig, ax, mesh, H, xedges, yedges
    