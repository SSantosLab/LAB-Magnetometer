import csv
import os
import time
from daqhats import mcc128, OptionFlags, HatIDs, HatError, AnalogInputMode, AnalogInputRange
from daqhats_utils import select_hat_device, chan_list_to_mask
import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser(
                    prog='Big Mag 0.1',
                    description='Magnetometer finite time scan and record.',
                    epilog=     '-----------------------------------------')

parser.add_argument('savedir',type=str)
parser.add_argument('-t', '--time', help='record time (seconds)',type=float, default=10.0)
parser.add_argument('-s', '--scanrate', help='scan rate (S/second)',type=float,default=1000.0)
parser.add_argument('-d', '--dtype', help='data type: int/float. If float, data is calibrated; if int, data is raw daq code and not calibrated.',type=str,default='float')

READ_ALL_AVAILABLE = -1

def continuous_scan_givedata(channels, scan_rate, t_measure, dtype):
    """
    Perform continuous acquisition for t_measure seconds, return acquired data.
    
    Args:
        channels (list): List of channel indices.
        scan_rate (float): Sample rate in Hz.
        t_measure (float): Total measurement time in seconds.
    
    Returns:
        numpy.ndarray: shape (total_samples, num_channels)
    """
    channel_mask = chan_list_to_mask(channels)
    num_channels = len(channels)
    input_mode = AnalogInputMode.SE
    input_range = AnalogInputRange.BIP_10V
    samples_per_channel = 0

    options = OptionFlags.CONTINUOUS
    if dtype =='float':
        options = OptionFlags.CONTINUOUS
    elif dtype =='int':
        options = OptionFlags.CONTINUOUS | OptionFlags.NOCALIBRATEDATA | OptionFlags.NOSCALEDATA # uncalibrated unscaled data

    try:
        address = select_hat_device(HatIDs.MCC_128)
        hat = mcc128(address)
        hat.a_in_mode_write(input_mode)
        hat.a_in_range_write(input_range)

        actual_scan_rate = hat.a_in_scan_actual_rate(num_channels, scan_rate)
        print(f"Using actual scan rate: {actual_scan_rate} Hz")

        hat.a_in_scan_start(channel_mask, samples_per_channel, scan_rate, options)
        print("Scan started. Acquiring...")

        read_request_size = READ_ALL_AVAILABLE
        timeout = 5.0
        start_time = time.time()

        data_blocks = []  # list to store data chunks

        while True:
            read_result = hat.a_in_scan_read(read_request_size, timeout)

            if read_result.hardware_overrun:
                print("\nHardware overrun!")
                break
            if read_result.buffer_overrun:
                print("\nBuffer overrun!")
                break

            data_pack = np.array(read_result.data)
            if dtype == 'int':
                data_pack = data_pack.astype(np.int16)
            
            samples_read_per_channel = int(len(data_pack) / num_channels)
            
            # Reshape into (samples_read_per_channel, num_channels)
            data_block = data_pack.reshape((samples_read_per_channel, num_channels))
            
            data_blocks.append(data_block)

            if (time.time() - start_time) >= t_measure:
                print("Measurement complete.")
                break

        hat.a_in_scan_stop()
        hat.a_in_scan_cleanup()

        # Stack all blocks into one array
        if data_blocks:
            data = np.vstack(data_blocks)
        else:
            data = np.empty((0, num_channels))

        return data

    except (HatError, ValueError) as err:
        print("\n", err)
        return None

if __name__ == '__main__':
    # Example usage:
    
    args = parser.parse_args()
    file_path = args.savedir
    dtype = args.savedir
    channels = [0, 1, 4]
    scan_rate = args.scanrate
    t_measure = args.time
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    timestamp_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    filename = file_path+f"mag_{timestamp_str}.hdf5"

    data_arr = continuous_scan_givedata(channels, scan_rate, t_measure, dtype)

    with h5py.File(filename, "w") as f:
        dset = f.create_dataset("voltage", data=data_arr)
        dset.attrs['dtype'] = dtype
        dset.attrs['sample_rate'] = scan_rate
        dset.attrs['end_time'] = timestamp_str
        dset.attrs['measure_time'] = t_measure
        print(f"result saved to {filename}")
