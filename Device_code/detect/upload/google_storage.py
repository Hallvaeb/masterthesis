# for uploading images to cloud storage buckets on google
import os
from google.cloud import storage
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/home/pi/config/hallmonitorserviceaccount.json'

def upload_blob(bucket_name, source_file_name,  stream):
    """Uploads file to the bucket"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file_name)

    stream.seek(0)

    # Chunk size reduced due to cellular connection
    storage.blob._DEFAULT_CHUNKSIZE = 524288 # 512 * 1024 B * 2 = 512 KB
    storage.blob._MAX_MULTIPART_SIZE = 524288 # 512 KB

    blob.upload_from_file(stream, content_type="image/jpeg", timeout=60, num_retries=5)

    print(
        "File {} uploaded.".format(
            source_file_name
        )
    )

