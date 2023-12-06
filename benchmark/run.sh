##########################################
# Arima Autoformer FEDformer-w Informer Prophet Transformer
##########################################


##########################################
# Train Budget


# Models that require training
is_training=1

for model in Arima Autoformer FEDformer-w Informer Prophet Transformer 
do

for budget in 50 100 150 200 250 300 500 
do

for preLen in 6 8 14 18 24 36 48 60
do

# exchange
python run.py \
 --is_training $is_training \
 --data exchange \
 --root_path ../academic_data/exchange_rate/ \
 --data_path exchange_rate.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5 \

nvidia-smi

# illness
python run.py \
 --is_training $is_training \
 --data ili \
 --root_path ../academic_data/illness/ \
 --data_path national_illness.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5

nvidia-smi

# weather
python run.py \
 --is_training $is_training \
 --data weather-mean \
 --root_path ../academic_data/weather/ \
 --data_path weather_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5

nvidia-smi

# traffic
python run.py \
 --is_training $is_training \
 --data traffic-mean \
 --root_path ../academic_data/traffic/ \
 --data_path traffic_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5

nvidia-smi

# electricity
python run.py \
 --is_training $is_training \
 --data ECL-mean \
 --root_path ../academic_data/electricity/ \
 --data_path electricity_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5

nvidia-smi

# ETTh1
python run.py \
 --is_training $is_training \
 --data ETTh1-mean \
 --root_path ../academic_data/ETT-small/ \
 --data_path ETTh1_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5

nvidia-smi

# ETTh2
python run.py \
 --is_training $is_training \
 --data ETTh2-mean \
 --root_path ../academic_data/ETT-small/ \
 --data_path ETTh2_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5

nvidia-smi

done;

done;

done;




##########################################
# Time Budget

# time-based budget
is_training=1

for model in Arima Autoformer FEDformer-w Informer Prophet Transformer 
do

for time_budget in 1 5 10 15 30 45 60 120
do

for preLen in 6 8 14 18 24 36 48
do

# exchange
python run.py \
 --is_training $is_training \
 --data exchange \
 --root_path ../academic_data/exchange_rate/ \
 --data_path exchange_rate.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --time_budget $time_budget \
 --itr 5 \

nvidia-smi


# illness
python run.py \
 --is_training $is_training \
 --data ili \
 --root_path ../academic_data/illness/ \
 --data_path national_illness.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --time_budget $time_budget \
 --itr 5

nvidia-smi

# weather
python run.py \
 --is_training $is_training \
 --data weather \
 --root_path ../academic_data/weather/ \
 --data_path weather_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --time_budget $time_budget \
 --itr 5

nvidia-smi

# traffic
python run.py \
 --is_training $is_training \
 --data traffic \
 --root_path ../academic_data/traffic/ \
 --data_path traffic_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --time_budget $time_budget \
 --itr 5

nvidia-smi

# electricity 
python run.py \
 --is_training $is_training \
 --data ECL \
 --root_path ../academic_data/electricity/ \
 --data_path electricity_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --time_budget $time_budget \
 --itr 5

nvidia-smi

# ETTh1
python run.py \
 --is_training $is_training \
 --data ETTh1 \
 --root_path ../academic_data/ETT-small/ \
 --data_path ETTh1_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --time_budget $time_budget \
 --itr 5

nvidia-smi

# ETTh2
python run.py \
 --is_training $is_training \
 --data ETTh2 \
 --root_path ../academic_data/ETT-small/ \
 --data_path ETTh2_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --time_budget $time_budget \
 --itr 5

nvidia-smi

done;

done;

done;







##########################################
# ForecastPFN
##########################################

is_training=0
model=ForecastPFN

for budget in 50 #100 150 200 250 300 500  
do

for preLen in 6 8 14 18 24 36 48
do


# exchange
python run.py \
 --is_training $is_training \
 --data exchange \
 --root_path ../academic_data/exchange_rate/ \
 --data_path exchange_rate.csv \
 --model $model \
 --model_path ../saved_weights \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5 \

nvidia-smi


# illness
python run.py \
 --is_training $is_training \
 --data ili \
 --root_path ../academic_data/illness/ \
 --data_path national_illness.csv \
 --model $model \
 --model_path ../saved_weights \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5

nvidia-smi


# weather
python run.py \
 --is_training $is_training \
 --data weather \
 --root_path ../academic_data/weather/ \
 --data_path weather_agg.csv \
 --model $model \
 --model_path ../saved_weights \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5

nvidia-smi


# traffic
python run.py \
 --is_training $is_training \
 --data traffic \
 --root_path ../academic_data/traffic/ \
 --data_path traffic_agg.csv \
 --model $model \
 --model_path ../saved_weights \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5

nvidia-smi


# electricity
python run.py \
 --is_training $is_training \
 --data ECL \
 --root_path ../academic_data/electricity/ \
 --data_path electricity_agg.csv \
 --model $model \
 --model_path ../saved_weights \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5

nvidia-smi


# ETTh1
python run.py \
 --is_training $is_training \
 --data ETTh1 \
 --root_path ../academic_data/ETT-small/ \
 --data_path ETTh1_agg.csv \
 --model $model \
 --model_path ../saved_weights \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5

nvidia-smi


# ETTh2
python run.py \
 --is_training $is_training \
 --data ETTh2 \
 --root_path ../academic_data/ETT-small/ \
 --data_path ETTh2_agg.csv \
 --model $model \
 --model_path ../saved_weights \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5

nvidia-smi

done;

done;












##########################################
# SeasonalNaive Mean Last
##########################################

is_training=0
for model in SeasonalNaive Mean Last 
do

for budget in 50 #100 150 200 250 300 500  
do

for preLen in 6 8 14 18 24 36 48 60
do

# exchange
python run.py \
 --is_training $is_training \
 --data exchange \
 --root_path ../academic_data/exchange_rate/ \
 --data_path exchange_rate.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5 \
 --metalearn_freq D

nvidia-smi


# illness
python run.py \
 --is_training $is_training \
 --data ili \
 --root_path ../academic_data/illness/ \
 --data_path national_illness.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5 \
 --metalearn_freq W

nvidia-smi


# weather
python run.py \
 --is_training $is_training \
 --data weather \
 --root_path ../academic_data/weather/ \
 --data_path weather_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5 \
 --metalearn_freq D

nvidia-smi


# traffic
python run.py \
 --is_training $is_training \
 --data traffic \
 --root_path ../academic_data/traffic/ \
 --data_path traffic_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5 \
 --metalearn_freq D

nvidia-smi


# electricity
python run.py \
 --is_training $is_training \
 --data ECL \
 --root_path ../academic_data/electricity/ \
 --data_path electricity_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5 \
 --metalearn_freq D

nvidia-smi


# ETTh1
python run.py \
 --is_training $is_training \
 --data ETTh1 \
 --root_path ../academic_data/ETT-small/ \
 --data_path ETTh1_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5 \
 --metalearn_freq D

nvidia-smi




# ETTh2
python run.py \
 --is_training $is_training \
 --data ETTh2 \
 --root_path ../academic_data/ETT-small/ \
 --data_path ETTh2_agg.csv \
 --model $model \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --train_budget $budget \
 --itr 5 \
 --metalearn_freq D

nvidia-smi


done;

done;

done;
