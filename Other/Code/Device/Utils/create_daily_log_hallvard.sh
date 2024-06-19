#!/bin/bash

path="/home/pi/logs_hallvard"

filename=hm40-$(date +%d-%m-%Y).txt

full_path="$path/$filename"

touch "$full_path"

chmod 755 "$full_path"

echo "File $full_path created to contain todays logs."