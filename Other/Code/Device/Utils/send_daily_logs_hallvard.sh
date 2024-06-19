#!/bin/bash

echo "Running send_daily_logs.sh"

dir="/greengrass/v2/packages/artifacts-unarchived/com.hm3.03.detect_hallvard/1.0.4/detect/utils"

command="sudo python upload_log_hallvard.py"

echo "Running command $command" 
echo "In: $dir"
cd "$dir" || exit 1
eval "$command"

echo "send_daily_logs.sh finished."