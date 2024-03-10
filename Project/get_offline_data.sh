#!/bin/bash

echo -e ">>> Get ALL Offline RL Data (downloading and parsing trajectories)\n\n"

LOG_DIR="log"
mkidr -p "${LOG_DIR}"

python3 get_offline_data.py --task "download" > "${LOG_DIR}/download_all.log"

parse_nohup(){
  data=$1
  level=$2
  echo -e ">>> Parse: ${data}-${level} <<<"
  nohup python3 get_offline_data.py --task "parse" --data "${data}" --level "${level}" > \
    "${LOG_DIR}/${data}-${level}.log" 2>&1 &
}

for cur_data in "halfcheetah" "hopper" "walker2d" "ant"; do
  for cur_level in "random" "medium" "expert" "medium-replay" "medium-expert"; do
    parse_nohup "${cur_data}" "${cur_level}"
  done
done

echo -e "\n\n>>> DONE ALL <<<\n\n"
