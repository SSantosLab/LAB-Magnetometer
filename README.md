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
## CheatSheet
1. Start Screen session:
```
screen -S <session_name>
```

2. Environment activation:
```
cd gdzhao
source .env/bin/activate
```
3. Start Acquisition (each in individual screen sessions):

Monitor raspberry-pi health using `python monitorpi.py -s <scan_rate> <save_dir>`. A recommendation is `-s 0.5` use default. Saves in save_dir/logs/ directory.

Scan magnetometer using `python scan_save_rawh5_fault_tolerant.py -s <scan_rate> -t <t_measure> <save_dir>`.

## Scan to binary files
- `scan_save_rawh5.py` saves one hdf5 file with continuous writing. Data is raw adc code without calibration in uint 16 format.
- `scan_save_rawh5_fault_tolerant.py` scan at given scanrate, by default dumps hdf5 file parts every 1 hour. Data is raw adc code in uint16 format.

## Scan to save (old)
- `continuous_scan_save.py` saves with either hdf5 or csv. The acquisition length should **not exceed the memory limit** of the raspberry-pi.
- `continuous_scan_saveh5.py` saves with hdf5 only. The acquisition length should **not exceed the memory limit** of the raspberry-pi.
- `continuous_scan_savecsv.py` saves line by line the readout. Has low memory usage and *can handle longer durations*.

## Raspberry-pi health logs
- `monitorpi.py` saves a csv of cpu temperature, external voltage, inbox temperature, inbox humidity, inbox pressure.

## Spectral Analysis
- `magnetofft.py` uses fft to compute fft amplitude. New version of `magnetofft.py` contain Power Spectrum (PS) and Power Spectral Density (PSD) calculation from [FFT_report](https://holometer.fnal.gov/GH_FFT.pdf). Also contains plotting routines for PS and PSD.
