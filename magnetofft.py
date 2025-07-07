import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import h5py
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
import polars as pl

#################################################
# Loading data
#################################################
plt.rcParams.update({'font.size': 14})

def load_hdf5(pathh5):
    '''
    load hdf5 and extract metadata, convert into magnetic field value in μT
    '''
    f = h5py.File(pathh5,'r')
    data = f['voltage']
    dset = {}

    if data.ndim == 1 or data.shape[1] == 1:
        # Single-channel case
        dset['x'] = data.flatten()
    else:
        # Multi-channel case, assuming channel order 0, 1, 4 -> x, z, y
        dset['x'] = data[:, 0]
        dset['y'] = data[:, 2]
        dset['z'] = data[:, 1]   

    dset['sample_rate'] = data.attrs['sample_rate']
    dset['measure_time'] = data.attrs['measure_time']
    dset['end_time'] = data.attrs['end_time']

    return dset

def load_csv_pl(pathcsv):
    df = pl.read_csv(pathcsv)
    data = df.to_numpy()
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

def fft_asd(chs,samplerate):
    freq, real, imag = fft_timeseries(chs, samplerate)
    power_spectrum = real**2 + imag**2
    
    dt = 1. / samplerate
    N = len(chs)
    psd = power_spectrum * dt / N / 2. / np.pi  # Convert to power spectral density
    asd = np.sqrt(psd)
    return freq, asd

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

def compute_averaged_psd(chs,fs,Lbin:int,overlapratio:float=0.5,nodc=True):
    Nhop = int(Lbin*overlapratio)
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
    return f,avg_psd

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
        ax.set_ylabel('FFT Amplitude[$\mu T$]')
    else:
        fig=None
        
    for vec in orientation:
        print('->plotting direction '+vec,end='\r')
        data = dset[vec]
        f,ps = compute_ps(data,fs)
        ax.loglog(f,np.sqrt(ps),label=label+vec,alpha=alpha)
        
    print('plot complete '+path[-20:-4]+'.')
    return fig,ax

def plot_sample_psd(path,fs=1000.,ax=None,label=None,orientation=['x','y','z'],Lbin=None,overlap=0.5,alpha=0.6):
    print('plot sample '+path+':')
    
    if path.endswith('.hdf5'):
        print('->loading hdf5...',end='\r')
        dset = load_hdf5(path)
    if path.endswith('.csv'):
        print('->loading csv ...',end='\r')
        dset = load_csv_pl(path)
    print('->load complete   ',end='\r')

    if Lbin is None:
        print('! Warning, no Lbin, taken default')
        N = len(dset['x'])
        Lbin = N//20
    
    if ax is None:
        fig,ax = plt.subplots(figsize = (8,6))
        ax.set_xlabel('f[Hz]')
        ax.set_ylabel('LSD[$\mu T/ \sqrt{Hz}$]')
    else:
        fig=None
    
    for vec in orientation:
        print('->plotting direction '+vec,end='\r')
        data = dset[vec]
        f,ps = compute_averaged_psd(data,fs,Lbin,overlap)
        ax.loglog(f,np.sqrt(ps),label=vec,alpha=alpha)
        
    print('plot complete '+path+'.')
    return fig,ax
    