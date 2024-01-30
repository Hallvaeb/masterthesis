import datetime
import google.cloud.logging
import logging
import sys
from google.cloud.logging.handlers import CloudLoggingHandler
sys.path.append('/home/pi/config')

import config


# write message to log file
# message: string
def write_to_log(message, publish_to_cloud=False):
  """Logging to cloud"""
  try:
    if publish_to_cloud:
      try:
        client = google.cloud.logging.Client()
        print(config.SERIAL_NR)

        hm3_handler = CloudLoggingHandler(client, name=config.SERIAL_NR)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if logger.hasHandlers() is False:
          #print("commented out here")
          logger.addHandler(hm3_handler)

        logging.info("[DETECTION] - " + message)
      except Exception as e:
        logging.error("Exception caught while trying to log to the cloud")
    now = datetime.datetime.now()
    date_time = now.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    date = now.strftime("%d-%b-%Y")
    with open("../logs/detect_yolo/yolo-" + str(date) + ".txt", "a") as f:
      f.write(date_time + ': ' + message + '\n')
  except Exception as e:
    logging.error("Exception caught while trying to write to log")
