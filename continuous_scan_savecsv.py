import csv
import os
import time
from daqhats import mcc128, OptionFlags, HatIDs, HatError, AnalogInputMode, AnalogInputRange
from daqhats_utils import select_hat_device, chan_list_to_mask
import argparse

parser = argparse.ArgumentParser(
                    prog='Big Mag 0.1',
                    description='Magnetometer finite time scan and record.',
                    epilog=     '-----------------------------------------')

parser.add_argument('savedir',type=str)
parser.add_argument('-t', '--time', help='record time (seconds)',type=float, default=10.0)
parser.add_argument('-s', '--scanrate', help='scan rate (S/second)',type=float,default=1000.0)

READ_ALL_AVAILABLE = -1

def continuous_scan_save(file_path, channels, scan_rate, t_measure):
    """
    Perform continuous acquisition for t_measure seconds, save to file_path.
    
    Args:
        file_path (str): Path to save CSV file.
        channels (list): List of channel indices.
        scan_rate (float): Sample rate in Hz.
        t_measure (float): Total measurement time in seconds.
    """
    channel_mask = chan_list_to_mask(channels)
    num_channels = len(channels)
    input_mode = AnalogInputMode.SE
    input_range = AnalogInputRange.BIP_10V
    samples_per_channel = 0
    options = OptionFlags.CONTINUOUS
    

    try:
        address = select_hat_device(HatIDs.MCC_128)
        hat = mcc128(address)
        hat.a_in_mode_write(input_mode)
        hat.a_in_range_write(input_range)

        actual_scan_rate = hat.a_in_scan_actual_rate(num_channels, scan_rate)
        print(f"Using actual scan rate: {actual_scan_rate} Hz")

        hat.a_in_scan_start(channel_mask, samples_per_channel, scan_rate, options)
        print("Scan started. Acquiring...")

        total_samples_read = 0
        read_request_size = READ_ALL_AVAILABLE
        timeout = 5.0

        start_time = time.time()
        
        with open(file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"Channel_{ch}" for ch in channels])

            while True:
                read_result = hat.a_in_scan_read(read_request_size, timeout)

                if read_result.hardware_overrun:
                    print("\nHardware overrun!")
                    break
                if read_result.buffer_overrun:
                    print("\nBuffer overrun!")
                    break

                samples_read_per_channel = int(len(read_result.data) / num_channels)

                for sample_index in range(samples_read_per_channel):
                    row = []
                    current_sample = total_samples_read + sample_index + 1
                    base = sample_index * num_channels
                    
                    for i in range(num_channels):
                        row.append(read_result.data[base + i])
                    
                    writer.writerow(row)

                total_samples_read += samples_read_per_channel

                # Check elapsed time
                if (time.time() - start_time) >= t_measure:
                    print("Measurement complete.")
                    break

        hat.a_in_scan_stop()
        hat.a_in_scan_cleanup()
        print(f"Data saved to {file_path}")

    except (HatError, ValueError) as err:
        print("\n", err)


if __name__ == '__main__':
    # Example usage:
    
    args = parser.parse_args()
    file_path = args.savedir

    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    timestamp_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    filename = file_path+f"mag_{timestamp_str}.csv"
    
    channels = [0]             # channels to record only 0 for shortcircuit!!
    scan_rate = args.scanrate
    t_measure = args.time

    continuous_scan_save(filename, channels, scan_rate, t_measure)
