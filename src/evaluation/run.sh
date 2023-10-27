#!bin/bash

i=0
while [ $i -lt 1000 ]
do
    python evaluate.py -c config.example.yaml >> /home/ubuntu/notebooks/forecasting/pretraining/logs/fixed_transformer_weibull_60_30_10_eval.log
    ((i=i+1))
    sleep 60
done