import subprocess
import time
import os

# === CONFIGURATION ===
SCRIPT_NAME = "scan_save_rawh5_fault_tolerant.py"
SCRIPT_PATH = "/home/gw-group/gdzhao/scan_save_rawh5_fault_tolerant.py"
SAVE_DIR = "/home/gw-group/gdzhao/acquisitions/"
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

def start_script():
    """Start the acquisition script in a detached screen session."""
    cmd = f"python3 {SCRIPT_PATH} -t {DURATION} -s {SAMPLERATE} {SAVE_DIR}"
    subprocess.call(["screen", "-dmS", SCREEN_SESSION, "bash", "-c", cmd])
    print("New DAQ script started.")

if __name__ == "__main__":
    while True:
        if not is_script_running():
            time_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
            print("DAQ script exit detected at {}. Restarting...".format(time_str))
            if is_screen_running():
                subprocess.call(["screen", "-S", SCREEN_SESSION, "-X", "quit"])
            start_script()
        else:
            print("DAQ script is running.",end='\r')
        time.sleep(60)  # check every minute
