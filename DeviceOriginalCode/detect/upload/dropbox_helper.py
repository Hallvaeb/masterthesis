import dropbox
import uuid
from datetime import datetime
import sys
sys.path.append('/home/jetsonnano3/hallmonitor/hm3/configs')
#from config import DROPBOX_KEY


def upload_image(imageBuffer, folderName):
    """

    @param imageBuffer: in-memory byte-image
    @param folderName: folder name on dropbox
    """
    print("Uploading image..")
    #dbx = dropbox.Dropbox(DROPBOX_KEY)
    filename = generate_name()
    #dbx.files_upload(
       # imageBuffer.getvalue(), "/hallmonitor/" + folderName + "/" + filename
    #)


def generate_name():
    now = datetime.now()
    current_time = now.strftime("%d-%m-%y_%H:%M:00" + str(uuid.uuid4()))
    return str(current_time) + ".jpg"
