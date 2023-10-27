#!/usr/bin/env bash

DOCKER_IMAGE_NAME=$2

if [[ "$(docker images -q ${DOCKER_IMAGE_NAME} 2> /dev/null)" == "" ]]; then
  docker build . -t ${DOCKER_IMAGE_NAME}
  docker push ${DOCKER_IMAGE_NAME}
fi

project_dir=$(dirname $(pwd))
pushd ../$1 > /dev/null

i=0
for model_dir in `ls -d */`; do
  model=${model_dir%%/}
  job_id=$(cluster submit --name=$1/${model} --image=${DOCKER_IMAGE_NAME} -v ${project_dir}:/project -w \
  /project/source -e PYTHONPATH=/project/source --cpu=1 --gpu=1 --mem=16 --restartable -- bash -c \
  "`cat ${model}/experiment.cmd` >> /project/$1/${model}/experiment.log 2>&1")

  let i=i+1
  echo "[${i}] Submitted $1/${model} with job id: ${job_id}"
done