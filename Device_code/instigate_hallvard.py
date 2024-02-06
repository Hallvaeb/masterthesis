#!/usr/bin/env python

import os

from subprocess import run, PIPE

# Give access to video on ggc_user
result = run(
    ["sudo", "usermod", "-a", "-G", "video", "ggc_user"],
    stdout=PIPE,
    stderr=PIPE,
    check=True,
)
print(result.stdout.decode())

# Create Reboot At Night bash script
bash_reboot_at_night_path = "/home/ggc_user/reboot_at_night.sh"

if not os.path.exists(bash_reboot_at_night_path):
    bash_reboot_at_night_content = """\
#!/bin/bash

# Define the file name and content
file_name="reboot_at_night_textfile"
file_content="Rebooted"

# Create the file in the current directory and write content to it
echo "$file_content" > "$file_name"

# Reboot the Raspberry Pi
sudo reboot
"""

    # Write the Bash script content to the file
    with open(bash_reboot_at_night_path, "w") as file:
        file.write(bash_reboot_at_night_content)

    # Provide executable rights to the Bash script
    os.chmod(bash_reboot_at_night_path, 0o755)

# Create startHallMonitor detect bash script 
bash_script_path = "/home/ggc_user/startHallMonitor.sh"

# Create/overwrite the Bash script
bash_script_content = """\
#!/bin/bash

# Directory to search
directory="/greengrass/v2/packages/artifacts-unarchived/com.hm3.03.hallvard_detect"

# Command to run
command="python detect_yolo.py --pi --confidence 60 --camflip 2"

# Get list of directories matching version pattern
directories=$(find "$directory" -type d -regex '.*/[0-9]+\.[0-9]+\.[0-9]+')

# Initialize variables
highest_version=""
highest_major=0
highest_minor=0
highest_patch=0

# Iterate over each directory
for dir in $directories; do
    # Extract version components
    version=$(basename "$dir")
    major=$(cut -d. -f1 <<< "$version")
    minor=$(cut -d. -f2 <<< "$version")
    patch=$(cut -d. -f3 <<< "$version")

    # Compare version components
    if ((major > highest_major)) || ((major == highest_major && minor > highest_minor)) || ((major == highest_major && minor == highest_minor && patch > highest_patch)); then
        highest_version="$version"
        highest_major=$major
        highest_minor=$minor
        highest_patch=$patch
    fi
done

# Extract the value of DEVICE_ENABLED from config.py
DEVICE_ENABLED=$(grep 'DEVICE_ENABLED' /home/pi/config/config.py | cut -d '=' -f 2 | sed 's/^[  ]*//;s/[        ]*$//' | tr -d '"')

# Check if DEVICE_ENABLED is set to TRUE
if [[ "$DEVICE_ENABLED" == "TRUE" ]]; then
    # Run the command in the highest version directory
    if [[ -n "$highest_version" ]]; then
        echo "Running command in the highest version directory: $highest_version"
        cd "$directory/$highest_version/detect" || exit 1
        eval "$command"
    else
        echo "No version directories found in $directory"
    fi
else
    echo "Device is not enabled. Set DEVICE_ENABLED to TRUE in config.py to run detect_yolo."
fi
"""

# Check if the bash script exists
if os.path.exists(bash_script_path):
    # Open the file and check if it contains the word DEVICE_ENABLED
    with open(bash_script_path, 'r') as file:
        script_content = file.read()
        if 'DEVICE_ENABLED' not in script_content:
            # If the word DEVICE_ENABLED is not in file, overwrite the file
            with open(bash_script_path, 'w') as file:
                file.write(bash_script_content)
            print("Bash script overwritten with DEVICE_ENABLED content.")
        else:
            print("Bash script exists AND contains DEVICE_ENABLED.")
else:
    # If the bash script doesn't exist, create a new one
    with open(bash_script_path, 'w') as file:
        file.write(bash_script_content)
    # Provide executable rights to the Bash script
    os.chmod(bash_script_path, 0o755)
