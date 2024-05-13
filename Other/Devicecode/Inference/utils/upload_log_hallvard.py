# for uploading images to cloud storage buckets on google
import os
from google.cloud import storage
import datetime
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/home/pi/config/hallmonitorserviceaccount.json'
from logger_hallvard import LoggerHallvard
import time

def upload_logs_as_blob():
    """Uploads logs_hallvard file from today to the bucket"""
    try: 
        log = LoggerHallvard(time.time())
    except Exception as e:
        print("failed to make LoggerHallvard:", e)
    
    try:
        # Bucket_name and filename
        # filename = "hm40-01-05-2024.txt"
        filename = "hm40-"+str(datetime.datetime.now().strftime("%d-%m-%Y")) + ".txt"
        full_path= "/home/pi/logs_hallvard/" + filename
        bucket_name = "daily_logs_hallvard"

        # Mostly same as upload_blob()
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(filename)

        with open(full_path, 'rb') as stream:
            # Chunk size reduced due to cellular connection
            storage.blob._DEFAULT_CHUNKSIZE = 524288  # 512 * 1024 B * 2 = 512 KB
            storage.blob._MAX_MULTIPART_SIZE = 524288  # 512 KB

            blob.upload_from_file(stream, content_type="text/plain", timeout=60, num_retries=5)

        log.write(f"Log file {full_path} uploaded.")
    except Exception as e:
        print("Failed to upload logs_hallvard as blob:", e)

if __name__ == "__main__":
    upload_logs_as_blob()