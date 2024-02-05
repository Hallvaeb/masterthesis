import sys
import hashlib
import os
import requests
sys.path.append('..')
from configs import config

SENSORS = ['AO','CPU','GPU','PLL','PMIC', 'THERMAL']

def post_log(text):
    device_id = hashlib.sha256((config.SERIAL+config.TOKEN).encode('utf-8')).hexdigest()
    temperatures = get_temperatures()
    temp_log = format_temperature_output_string(temperatures)

    if(temp_log is not None):
        data = {
            "key": config.API_KEY,
            "device_id": device_id,
            "text": temp_log
        }
        response = requests.post(url=config.API_LOG_ENDPOINT, data=data)

def get_temperatures():
    output = os.popen('cat /sys/devices/virtual/thermal/thermal_zone*/temp').read()
    output = output.splitlines()
    print(str(output))
    return output

def format_temperature_output_string(temperatures: []):
    if (temperatures is None):
        return None
    temp_log = ''
    for i in range(len(temperatures)):
        temp_celsius = int(temperatures[i])/1000
        sensors_name = 'unknown'
        if (i < len(SENSORS)):
            sensors_name = SENSORS[i]
        temp_log += sensors_name + ': ' + str(temp_celsius) + 'C. '
        if (i < len(temperatures) - 1):
            temp_log += '\n'
    return temp_log


post_log("sent from device")
