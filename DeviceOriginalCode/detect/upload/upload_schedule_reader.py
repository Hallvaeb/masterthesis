import os
path = "/home/pi/config/upload_schedule"

#reads next index in schedule to see if next image should be uploaded
# returns true if it is time to upload. otherwise returns false
def should_upload_image():
    lines = None
    with open(path, "r") as f:
        lines = f.readlines()
    current_schedule_index = int(lines[0].strip())
    schedule = lines[1].strip()
    value = schedule[current_schedule_index]
    result = False
    if value == '1':
        result = True
    
    # get next value
    next_index = get_next_index(current_schedule_index, len(schedule))
    
    overwrite_schedule_file(next_index, schedule)
    return result

def get_next_index(index, length):
    print("index, length", index, length)
    if index >= length-1:
        print("returning 0")
        return 0
    else:
        index += 1
        print("returning: ", index)
        return index

# creates the file if it doenst exist
def overwrite_schedule_file(index ,schedule):
    with open(path, "w") as f:
        f.write(str(index) + '\n')
        f.write(schedule)

def should_upload_image_timebased(current_minutes):

    images_send_in_hour = 4
    hour_minutes = 60
    frequency = 12
    send_picture_at = (hour_minutes / frequency) * (frequency / images_send_in_hour)
    if int(current_minutes - (current_minutes % send_picture_at)) == current_minutes:
        return True
    else:
        return False
