#!/bin/bash

echo -e ">>> Get ALL Offline RL Data (downloading and parsing trajectories)\n\n"

LOG_DIR="log/get_offline_data"
mkdir -p "${LOG_DIR}"

conda activate 533v
python3 get_offline_data.py --task "download" > "${LOG_DIR}/download_all.log"

parse_nohup(){
  env=$1
  level=$2
  echo -e ">>> Parse: ${env}-${level} <<<"
  nohup python3 get_offline_data.py --task "parse" --env "${env}" --level "${level}" > \
    "${LOG_DIR}/${env}-${level}.log" 2>&1 &
}

for cur_env in "halfcheetah" "hopper" "walker2d" "ant"; do
  for cur_level in "random" "medium" "expert" "medium-replay" "medium-expert"; do
    parse_nohup "${cur_env}" "${cur_level}"
  done
done

echo -e "\n\n>>> DONE ALL <<<\n\n"
