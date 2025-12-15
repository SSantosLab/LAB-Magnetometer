import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
import matplotlibstyle
from magnetofft import *

def plot_single_psd(filepath,outpath,sample_rate=20000.0):
    end_time, _ = read_hdf5_times(filepath)
    fig,ax = plot_sample_psd(filepath,fs=sample_rate,orientation=['x','y','z'],label='',Lbin = 1000000)
    ax.legend(loc='lower left')
    ax.set_title(label = f'{end_time.year}:{end_time.month}:{end_time.day}:{end_time.hour}:{end_time.minute}')
    fig.savefig(outpath)
    plt.close()
    return None

def file_name_extract(filedir,outdir,filehead=None):
    filename_list = os.listdir(filedir)
    if filehead is None:
        filepath_list = [os.path.join(filedir,filename) for filename in filename_list]
        outpath_list = [os.path.join(outdir,filename[:-5]) for filename in filename_list]
        return filepath_list,outpath_list
    else:
        filepath_list = [os.path.join(filedir,filename) for filename in filename_list if filehead in filename]
        outpath_list = [os.path.join(outdir,filename[:-5]+'.png') for filename in filename_list if filehead in filename]
        return filepath_list,outpath_list

# def main():
#     filedir = '/data/atom-interferometer/2025_MagneticBackground/20250728_BedrettoCampaing/20250729_MagneticField/'
#     outdir = 'results/20250729_MagneticField/psd/'
#     corrupted_files_list = [
#         '/data/atom-interferometer/2025_MagneticBackground/20250728_BedrettoCampaing/20250729_MagneticField/mag_2025_08_14_11_21_part0.hdf5',
#         '/data/atom-interferometer/2025_MagneticBackground/20250728_BedrettoCampaing/20250729_MagneticField/mag_2025_08_07_18_18_part136.hdf5',
#         '/data/atom-interferometer/2025_MagneticBackground/20250728_BedrettoCampaing/20250729_MagneticField/mag_2025_07_31_06_38_part2.hdf5'
#     ]
#     if not os.path.exists(outdir):
#         os.makedirs(outdir, exist_ok=True)
#     filepath_list,outpath_list = file_name_extract(filedir,outdir,filehead='mag')
#     for filepath,outpath in zip(filepath_list,outpath_list):
#         if os.path.exists(outpath):
#             continue
#         elif filepath in corrupted_files_list:
#             continue
#             # corrupted file at powerdown
#         else:
#             print(filepath)
#             plot_single_psd(filepath,outpath,sample_rate=20000.0)

def main():
    filedir = '/data/atom-interferometer/2025_MagneticBackground/20250728_BedrettoCampaing/20250722_zerogauss/'
    outdir = 'results/20250722_zerogauss/psd_avg/'
    corrupted_files_list = [
        '/data/atom-interferometer/2025_MagneticBackground/20250728_BedrettoCampaing/20250729_MagneticField/mag_2025_08_14_11_21_part0.hdf5',
        '/data/atom-interferometer/2025_MagneticBackground/20250728_BedrettoCampaing/20250729_MagneticField/mag_2025_08_07_18_18_part136.hdf5',
        '/data/atom-interferometer/2025_MagneticBackground/20250728_BedrettoCampaing/20250729_MagneticField/mag_2025_07_31_06_38_part2.hdf5',
        '/data/atom-interferometer/2025_MagneticBackground/20250728_BedrettoCampaing/20250722_zerogauss/mag_2025_07_23_06_31_part34.hdf5'
    ]
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    filepath_list,outpath_list = file_name_extract(filedir,outdir,filehead='mag')
    for filepath,outpath in zip(filepath_list,outpath_list):
        if os.path.exists(outpath):
            continue
        elif filepath in corrupted_files_list:
            continue
            # corrupted file at powerdown
        else:
            print(filepath)
            save_averaged_ps_psd_hdf5(
                h5_in= filepath,
                out_dir = outdir,
                channels=('x','y','z'),
                Lbin = None,
                overlap = 0.5,
                nodc = True,
                fmax = None,
                fmin = 1e-2
            )

if __name__ == '__main__':
    main()