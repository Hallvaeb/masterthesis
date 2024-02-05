import os
import subprocess
import json
import hashlib
import e3372
from datetime import datetime, timezone
import sys

sys.path.append('/home/pi/config')
from config import API_KEY, SERIAL, TOKEN

SENSORS = ['AO', 'CPU', 'GPU', 'PLL', 'PMIC', 'THERMAL']


def __read_ram_usage():
  ram_usage = os.popen("free | awk 'FNR == 2 {print $3/1000}'").read()
  return ram_usage


def __read_current_cpu_utilization():
  cmd = "awk '{u=$2+$4; t=$2+$4+$5; if (NR==1){u1=u; t1=t;} else print ($2+$4-u1) * 100 / (t-t1); }' <(grep 'cpu ' /proc/stat) <(sleep 1;grep 'cpu ' /proc/stat)"
  output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, executable="/bin/bash").stdout.read()
  return output.decode()


def get_full_health_check():
  cpu_utilization = __read_current_cpu_utilization().strip()
  ram_usage = __read_ram_usage().strip()
  temperatures = get_temperatures_of_cpu_gpu()
  signal_values = json.loads(e3372.get_signal_values())

  DEVICE_ID = hashlib.sha256((SERIAL + TOKEN).encode('utf-8')).hexdigest()
  timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

  data = {
    "api_key": API_KEY,
    "device_id": DEVICE_ID,
    "timestamp": timestamp,
    "rsrq": signal_values['rsrq'],
    "rsrp": signal_values['rsrp'],
    "rssi": signal_values['rssi'],
    "sinr": signal_values['sinr'],
    "mode": signal_values['mode'],
    "cell_id": signal_values['cell_id'],
    "cpu_usage": cpu_utilization,
    "ram_usage": ram_usage,
    "cpu_temp": temperatures[0],
  }
  return data


def get_temperatures_of_cpu_gpu():
  output = os.popen('cat /sys/devices/virtual/thermal/thermal_zone*/temp').read()
  output = output.splitlines()

  return_list = []
  for i in range(len(output)):
    temp_celsius = int(output[i]) / 1000
    return_list.append(temp_celsius)
  return return_list


