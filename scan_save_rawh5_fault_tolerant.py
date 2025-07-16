#!/usr/bin/env python3
import os
import sys
import time
import signal
import argparse
import h5py
import numpy as np
from daqhats import mcc128, OptionFlags, HatIDs, HatError, AnalogInputMode, AnalogInputRange
from daqhats_utils import select_hat_device, chan_list_to_mask

# Global handles for cleanup
_HAT = None
_FILE = None
_DSET = None

# Constants
READ_ALL_AVAILABLE = -1
CHUNK_DURATION = 3600.0  # seconds per file
DEFAULT_CHUNKSIZE = 8192


def safe_fsync(h5file):
    """
    Attempt to fsync the underlying file descriptor of an HDF5 File.
    """
    try:
        fd = h5file.id.get_vfd_handle()
        os.fsync(fd)
    except Exception as e:
        print(f"Warning: fsync failed: {e}")


def on_exit(signum, frame):
    """
    Signal handler for graceful shutdown: stops acquisition,
    flushes current buffers, then exits to trigger main cleanup.
    """
    global _HAT, _FILE, _DSET
    print(f"Received signal {signum}, stopping acquisition...")
    try:
        if _HAT:
            _HAT.a_in_scan_stop()
            _HAT.a_in_scan_cleanup()
    except Exception:
        pass
    try:
        if _DSET:
            _DSET.flush()
        if _FILE:
            safe_fsync(_FILE)
    except Exception:
        pass
    # Raise SystemExit to unwind and run main finally block
    sys.exit(0)


def register_signal_handlers():
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGPWR):
        try:
            signal.signal(sig, on_exit)
        except (AttributeError, RuntimeError):
            # Some signals may not exist
            pass


def continuous_scan_with_rotation(channels, scan_rate, total_time, savedir, prefix, chunksize=DEFAULT_CHUNKSIZE):
    """
    Continuous acquisition with hourly HDF5 rotation, SWMR mode,
    and immediate flush+fsync for crash resilience.

    Args:
        channels (list[int]): channel indices
        scan_rate (float): samples per second
        total_time (float): total run time in seconds
        savedir (str): output directory
        prefix (str): filename prefix (e.g. 'mag_2025_07_16_12_00')
    """
    global _HAT, _FILE, _DSET

    os.makedirs(savedir, exist_ok=True)
    channel_mask = chan_list_to_mask(channels)
    num_channels = len(channels)
    options = OptionFlags.CONTINUOUS | OptionFlags.NOCALIBRATEDATA | OptionFlags.NOSCALEDATA
    input_mode = AnalogInputMode.SE
    input_range = AnalogInputRange.BIP_10V

    # Register cleanup handlers
    register_signal_handlers()

    # Connect to device
    address = select_hat_device(HatIDs.MCC_128)
    hat = mcc128(address)
    _HAT = hat
    hat.a_in_mode_write(input_mode)
    hat.a_in_range_write(input_range)
    actual_rate = hat.a_in_scan_actual_rate(num_channels, scan_rate)
    print(f"Using actual scan rate: {actual_rate} Hz")

    # Prepare rotation
    start_time = time.time()
    next_rotate = start_time + CHUNK_DURATION
    file_count = 0

    def open_new_file(idx):
        filename = os.path.join(savedir, f"{prefix}_part{idx}.hdf5")
        f = h5py.File(filename, "w", libver="latest", swmr=True)
        f.swmr_mode = True
        dset = f.create_dataset(
            "voltage", shape=(0, num_channels), maxshape=(None, num_channels),
            chunks=(chunksize, num_channels), dtype="uint16"
        )
        # Metadata
        dset.attrs['dtype'] = 'uint16'
        dset.attrs['sample_rate'] = scan_rate
        start_ts = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        dset.attrs['start_time'] = start_ts
        dset.attrs['measure_time'] = min(CHUNK_DURATION, total_time)
        # Pre-create end_time attribute for safe SWMR updates
        dset.attrs['end_time'] = start_ts  # placeholder
        return f, dset

    # Open initial file
    f, dset = open_new_file(file_count)
    _FILE, _DSET = f, dset
    buffer = []

    # Start scan
    hat.a_in_scan_start(channel_mask, 0, scan_rate, options)
    print("Scan started. Acquiring...")

    try:
        while True:
            # Read raw data
            result = hat.a_in_scan_read(READ_ALL_AVAILABLE, timeout=5.0)
            if result.hardware_overrun:
                print("\nHardware overrun!")
                break
            if result.buffer_overrun:
                print("\nBuffer overrun!")
                break

            block = np.array(result.data, dtype=np.uint16)
            rows = block.size // num_channels
            block = block.reshape((rows, num_channels))
            buffer.append(block)

            # Rotate file if needed
            now = time.time()
            if now >= next_rotate:
                # Flush current buffer
                if buffer:
                    combined = np.vstack(buffer)
                    old = dset.shape[0]
                    dset.resize((old + combined.shape[0], num_channels))
                    dset[old:, :] = combined
                    f.flush()
                    safe_fsync(f)
                    buffer.clear()
                # Close old file cleanly
                end_ts = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
                dset.attrs['end_time'] = end_ts
                f.flush()
                safe_fsync(f)
                f.close()

                # Next file
                file_count += 1
                next_rotate += CHUNK_DURATION
                f, dset = open_new_file(file_count)
                _FILE, _DSET = f, dset

            # Write if buffer large
            total_buf = sum(b.shape[0] for b in buffer)
            if total_buf >= chunksize:
                combined = np.vstack(buffer)
                old = dset.shape[0]
                dset.resize((old + combined.shape[0], num_channels))
                dset[old:, :] = combined
                f.flush()
                safe_fsync(f)
                buffer.clear()

            # Check total duration
            if now - start_time >= total_time:
                print("Measurement complete.")
                break

    except (HatError, ValueError) as err:
        print("\nError during acquisition:", err)

    finally:
        # Stop and cleanup
        hat.a_in_scan_stop()
        hat.a_in_scan_cleanup()
        # Final flush
        if buffer:
            combined = np.vstack(buffer)
            old = dset.shape[0]
            dset.resize((old + combined.shape[0], num_channels))
            dset[old:, :] = combined
        # Update end_time on final file
        end_ts = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        dset.attrs['end_time'] = end_ts
        f.flush()
        safe_fsync(f)
        f.close()
        print(f"Data saved to parts 0–{file_count} in {savedir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Big Mag 0.2',
        description='Magnetometer scan with hourly file rotation and crash resilience.'
    )
    parser.add_argument('savedir', type=str, help='Directory for output HDF5 files')
    parser.add_argument('-t', '--time', type=float, default=10.0,
                        help='Total record time in seconds')
    parser.add_argument('-s', '--scanrate', type=float, default=1000.0,
                        help='Requested scan rate (S/s)')
    args = parser.parse_args()

    ts = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    continuous_scan_with_rotation([0,1,4], args.scanrate, args.time,
                                  args.savedir, f"mag_{ts}")
