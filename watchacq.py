import subprocess
import time
import os
import argparse

# === CONFIGURATION ===
SCRIPT_NAME = "scan_save_rawh5_fault_tolerant.py"
SCRIPT_PATH = "/home/gw-group/gdzhao/scan_save_rawh5_fault_tolerant.py"
SAVE_DIR = "/home/gw-group/gdzhao/acquisitions/20250722_Bedretto/"
DURATION = "8640000" # 100 days by default
SAMPLERATE = "20000"
SCREEN_SESSION = "daq_session"

def is_script_running():
    """Check if the script is running."""
    try:
        output = subprocess.check_output(["pgrep", "-f", SCRIPT_NAME])
        return bool(output.strip())
    except subprocess.CalledProcessError:
        return False

def is_screen_running():
    """Check if the screen session exists."""
    try:
        output = subprocess.check_output(["screen", "-ls"], stderr=subprocess.DEVNULL).decode()
        return SCREEN_SESSION in output
    except Exception:
        return False

def start_script(savedir,tmeasure,samplerate):
    """Start the acquisition script in a detached screen session."""
    cmd = f"python3 {SCRIPT_PATH} -t {tmeasure} -s {samplerate} {savedir}"
    subprocess.call(["screen", "-dmS", SCREEN_SESSION, "bash", "-c", cmd])
    print("New DAQ script started.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Big Mag Watcher 1.0',
        description='Magnetometer scan watchdog.'
    )
    parser.add_argument('savedir', type=str, help='Directory for output HDF5 files')
    parser.add_argument('-t', '--time', type=float, default=8640000.0,
                        help='Total record time in seconds')
    parser.add_argument('-s', '--scanrate', type=float, default=20000.0,
                        help='Requested scan rate (S/s)')
    args = parser.parse_args()
    save_dir = args.savedir
    t_measure = args.time
    scan_rate = args.scanrate
    while True:
        if not is_script_running():
            time_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
            print("DAQ script exit detected at {}. Restarting...".format(time_str))
            if is_screen_running():
                subprocess.call(["screen", "-S", SCREEN_SESSION, "-X", "quit"])
            start_script(save_dir,t_measure,scan_rate)
        else:
            print("DAQ script is running.",end='\r')
        time.sleep(60)  # check every minute
