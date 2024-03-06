This directory is for the evaluation of ForecastPFN. We have evaluated ForecastPFN on seven real-world datasets that have been used in the literature. The datasets are in the `../academic_data` folder. The datasets include Illness, Exchange, ECL, ETTh1 and ETTh2, Weather and Traffic.

The evaluation has been done against multiple baselines which include Arima, Prophet, Informer, Fedformer-w, Autoformer, Transformer and Metalearn, as well as more simple baselines Mean, Last, and NaiveSeasonal.

This is an example where illness is evaluated on ForecastPFN:
```
# illness
python run.py \
 --is_training 0 \
 --data ili \
 --root_path ../academic_data/illness/ \
 --data_path national_illness.csv \
 --model ForecastPFN \
 --seq_len 36 \
 --label_len 18 \
 --pred_len 14 \
 --train_budget 50 \
 --itr 5
```

The arguments that are passed are:
- `is_training` : This is set to 0 for ForecastPFN and Metalearn since these models don't require training while it is set to 1 for all other models.
- `data` : This denotes which data should be used. Look at benchmark/data_provider/data_factory.py for more details.
- `root_path` : This denotes the parent directory which contains the required dataset.
- `data_path` : This denotes the name of the file which contains the data. Look into the academic_data folder for information regarding other dataset files.
- `model` : This is one of (ForecastPFN, Metalearn, Arima, Autoformer, Informer, Transformer, FEDformer-w, Prophet)
- `seq_len` : The length of the input sequence to be used. In our default setting, we have this set to 96 for exchange and 36 for all other datasets.
- `label_len` : In our default setting, we have this set to 48 for exchange and 18 for all other datasets.
- `pred_len` : This is the length of the prediction to be made. We have evaluated our model with various prediction lengths.
- `train_budget` : This denotes the number of training examples that are available to the models that they can use for training. ForecastPFN and Metalearn use 0 examples since they are zero-shot.
- `itr` : Number of times evaluation should be repeated. This affects the transformer-based models since they are non-deterministic.

All experiments that have been run for this paper can be found in `run.sh`. 

Replication of the paper tables and plots can be found in the jupyter notebook `./analyze_results.ipynb`.