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

# datatype is always int16 in this implementation

READ_ALL_AVAILABLE = -1

def continuous_scan_and_dump(channels, scan_rate, t_measure, filename, chunksize=8192):
    """
    Perform continuous acquisition for t_measure seconds, write buffered data into chuncks of a hdf5 file.datatype is set to int16 for compact storage.
    
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

    options = OptionFlags.CONTINUOUS | OptionFlags.NOCALIBRATEDATA | OptionFlags.NOSCALEDATA # uncalibrated daq raw code
    
    with h5py.File(filename,"w") as f:
        dset = f.create_dataset(
            "voltage", shape=(0, num_channels),
            maxshape=(None, num_channels),
            dtype='uint16',
            chunks=(chunksize, num_channels)
        )
        dset.attrs['dtype'] = 'int'
        dset.attrs['sample_rate'] = scan_rate
        dset.attrs['start_time'] = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        dset.attrs['measure_time'] = t_measure
        buffer = []
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

            while True:
                read_result = hat.a_in_scan_read(read_request_size, timeout)

                if read_result.hardware_overrun:
                    print("\nHardware overrun!")
                    break
                if read_result.buffer_overrun:
                    print("\nBuffer overrun!")
                    break

                block = np.array(read_result.data,dtype=np.uint16)
                samples_read_per_channel = int(len(block) / num_channels)

                block = block.reshape((samples_read_per_channel, num_channels))

                buffer.append(block)

                # flush when buffer large enough
                total = sum(b.shape[0] for b in buffer)
                if total >= chunksize:
                    combined = np.vstack(buffer)
                    old = dset.shape[0]
                    dset.resize((old + combined.shape[0], num_channels))
                    dset[old:,:] = combined
                    f.flush()
                    buffer = []

                if time.time() - start_time >= t_measure:
                    print("Measurement complete.")
                    break

            hat.a_in_scan_stop()
            hat.a_in_scan_cleanup()

            # final flush
            if buffer:
                combined = np.vstack(buffer)
                old = dset.shape[0]
                new = old + combined.shape[0]
                dset.resize((new, len(channels)))
                dset[old:new, :] = combined
                f.flush()

            dset.attrs['end_time'] = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
            f.flush()
            f.close()
            print(f"result saved to {filename}")
            return None
        
        except (HatError, ValueError) as err:
            print("\n", err)
            return None

if __name__ == '__main__':
    args = parser.parse_args()
    file_dir = args.savedir
    channels = [0, 1, 4]
    scan_rate = args.scanrate
    t_measure = args.time
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        
    timestamp_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    filename = file_dir+f"mag_{timestamp_str}.hdf5"

    continuous_scan_and_dump(channels,scan_rate,t_measure,filename)