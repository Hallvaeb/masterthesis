import psutil
import time
from logger import write_to_log
import os

def findProcessIdByName(processName):
  '''
    Get a list of all the PIDs of a all the running process whose name contains
    the given string processName
    '''
  listOfProcessObjects = []
  # Iterate over the all the running process
  for proc in psutil.process_iter():
    try:
      pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time', 'status', 'cmdline'])
      # Check if process name contains the given name string.
      if processName.lower() in pinfo['name'].lower():
        listOfProcessObjects.append(pinfo)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
      pass
  return listOfProcessObjects

def processKiller():
  current_pid = os.getpid()
  listOfProcessNames = ['python']
  for name in listOfProcessNames:
    listOfProcessIds = findProcessIdByName(name)

    if len(listOfProcessIds) > 0:
      for elem in listOfProcessIds:
        processID = elem['pid']
        if 'detect_yolo.py' in elem['cmdline'] and processID != current_pid:
          process = psutil.Process(processID)
          process.kill()
          write_to_log('-> detect_yolo killed.', True)