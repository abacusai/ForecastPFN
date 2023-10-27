#!bin/bash

i=0
while [ $i -lt 1000 ]
do
    python evaluate.py -c config.example.yaml >> /home/ubuntu/notebooks/forecasting/pretraining/logs/multiple_frequencies_eval.log
    ((i=i+1))
    sleep 120
done