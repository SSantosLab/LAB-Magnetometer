import adafruit_bme680
import time
import board
import argparse
import os
import csv
import subprocess

def read_voltage(channel="EXT5V_V"):
    try:
        output = subprocess.check_output(['vcgencmd', 'pmic_read_adc', channel], encoding='utf-8')
        voltage_str = output.split('=')[1].replace('V', '').strip()
        value = float(voltage_str)
        return value
    except Exception as e:
        return None

def read_cpu_temperature():
    try:
        output = subprocess.check_output(['vcgencmd', 'measure_temp'], encoding='utf-8')
        temp_str = output.strip().split('=')[1].replace("'C", "")
        return float(temp_str)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
                        prog='Raspberry-pi self health recorder',
                        description='Keep a Logging for raspberry-pi\n environment [temp, humidity,pressure]\n rpi-self [cputemp, 5vextvoltage].',
                        epilog=     '-----------------------------------------')
    
    parser.add_argument('savedir',type=str)
    parser.add_argument('-t', '--time', help='record time (seconds), if not passed, do continuous mode.',type=float, default=None)
    parser.add_argument('-s', '--scanrate', help='scan rate (S/min), default 1S/min',type=float,default=1)
    
    args = parser.parse_args()
    file_path = args.savedir + 'logs/'
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    timestamp_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    filename = file_path+f"monitorpi_{timestamp_str}.csv"
    
    interval = 60.0/args.scanrate
    t_measure = args.time
    
    print('Scan interval {}s'.format(interval))
    if t_measure is None:
        print('No scan time designated, assuming continuous... press ctrl+c to stop.')
    else:
        print('Scan time = {}s, start scanning...'.format(t_measure))
    
    # Create sensor object, communicating over the board's default I2C bus
    i2c = board.I2C()   # uses board.SCL and board.SDA
    bme680 = adafruit_bme680.Adafruit_BME680_I2C(i2c)
    
    # change this to match the location's pressure (hPa) at sea level
    bme680.sea_level_pressure = 1013.25
    
    time_start = time.time()
    
    try:
        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['localtime', 'extvolt(V)', 'cputemp(C)', 'envtemperature(C)', 'envhumidity(%)', 'envpressure(hPa)'])
            while True:
                if t_measure is not None and (time.time() - time_start) >= t_measure:
                    break
    
                timestamp_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
                extvolt = read_voltage()
                cputemp = read_cpu_temperature()
                temperature = bme680.temperature
                humidity = bme680.relative_humidity
                pressure = bme680.pressure
    
                writer.writerow([timestamp_str, extvolt, cputemp, temperature, humidity, pressure])
                csvfile.flush()
                time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping data logging...")
    
    print('data saved to'+ filename)

if __name__ == '__main__':
    main()