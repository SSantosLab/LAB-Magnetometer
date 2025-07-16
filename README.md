# LAB-Mangetometer
Code related to readout and analysis of data taken with a Bartington Magnetometer

# Requirements
Required submodule `daqhats` need to be installed in the virtual environment. See [DAQHATS](https://github.com/mccdaq/daqhats).

Other required libraries for spectral analysis include:
1. `dash` for web-ui.
2. `jupyter` for notebooks.
3. `polars` for large csv reading.
4. `h5py` for hdf5 file handling.
5. `scipy` for spectrograms.
6. `argparse` for python argument parsing.

# Scripts

## Scan to save
- `continuous_scan_save.py` saves with either hdf5 or csv. The acquisition length should **not exceed the memory limit** of the raspberry-pi.
- `continuous_scan_saveh5.py` saves with hdf5 only. The acquisition length should **not exceed the memory limit** of the raspberry-pi.
- `continuous_scan_savecsv.py` saves line by line the readout. Has low memory usage and *can handle longer durations*.

## Temperature and Humidity scan
- `temphumid.py` scans temperature humidity and air pressure every 2 seconds and print in terminal.

## Spectral Analysis
- `magnetofft.py` uses fft to compute fft amplitude. New version of `magnetofft.py` contain Power Spectrum (PS) and Power Spectral Density (PSD) calculation from [FFT_report](https://holometer.fnal.gov/GH_FFT.pdf). Also contains plotting routines for PS and PSD.
