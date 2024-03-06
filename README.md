# ForecastPFN

![alt text](img/forecastpfn.png?raw=true)

![Crates.io](https://img.shields.io/crates/l/Ap?color=orange)

This is the code repository for the paper [_ForecastPFN: Synthetically-Trained Zero-Shot Forecasting_](https://arxiv.org/abs/2311.01933). 

ForecastPFN is the first zero-shot forecasting model that is trained purely on synthetic data.
It is a prior-data fitted network, trained once, offline, which can make predictions on a new time series dataset in a single forward pass.
ForecastPFN is more accurate and faster compared to state-of-the-art forecasting methods, even when the other methods are allowed to train on hundreds of additional training data points.

The codebase has these parts: 
- `./src/` contains all code to replicate the ForecastPFN synthetic data generation and training procedure
- `./benchmark/` contains all the code to replicate the benchmark of ForecastPFN against the the other baselines. 

# Table of contents
1. [Installation](#installation-)
2. [Inference with pretrained model](#inference-with-pretrained-model-)
3. [Synthetic Data Generation](#synthetic-data-generation-)
4. [Model Training](#model-training-)

# Installation <a name="Installation"></a>
This repository uses Python 3.9.

You can start by first creating a new environment using `conda` or your preferred method.

```
# using conda
conda create -n fpfn python=3.9
conda activate fpfn
```

Next, install the dependencies in the `requirements.txt` file using,
```
pip install -r requirements.txt

```

Next, you can download the data used for our benchmark [here](https://drive.google.com/file/d/1-QujU6oKJ6cyFdSQ8uRA8PMdOdVaZBVL/view?usp=sharing). Make sure to put it in the folder `./academic_data/`.

Finally, the ForecastPFN model weights should be downloaded [here](https://drive.google.com/file/d/1acp5thS7I4g_6Gw40wNFGnU1Sx14z0cU/view?usp=sharing). Make sure to put it in the folder `./saved_weights/`.


# Inference with pretrained model <a name="Evaluation"></a>
We released our pretrained ForecastPFN model and saved it in this repository.
We evaluate ForecastPFN on the seven real-world datasets used in our paper (all of which are popular in recent forecasting literature). The datasets are Illness, Exchange, ECL, ETTh1, ETTh2, Weather and Traffic.
We compare ForecastPFN to multiple baselines such as Arima, Prophet, Informer, Fedformer-w, Autoformer, Transformer and Meta-N-Beats.

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
 --model_path ../saved_weights/ \
 --itr 5

The arguments that are passed are:
-- is_training : This is set to 0 for ForecastPFN and Metalearn since these models don't require training while it is set to 1 for all other models.
-- data : This denotes which data should be used. Look at benchmark/data_provider/data_factory.py for more details.
-- root_path : This denotes the parent directory which contains the required dataset.
-- data_path : This denotes the name of the file which contains the data. Look into the academic_data folder for information regarding other dataset files.
-- model : This is one of (ForecastPFN,Metalearn,Arima,Autoformer,Informer,Transformer,FEDformer-w,Prophet)
-- model_path : This is only required when evaluating ForecastPFN or Metalearn since these models are zero-shot
-- seq_len : The length of input sequence to be used. In our default setting, we have this set to 96 for exchange and 36 for all other datasets.
-- label_len : In our default setting, we have this set to 48 for exchange and 18 for all other datasets.
-- pred_len : This is the length of prediction to be made. We have evaluated our model with various prediction lengths.
-- train_budget : This denotes the number of training examples that are available to the models which they can use for training. ForecastPFN and Metalearn use 0 examples since they are zero-shot.
-- itr : Number of times evaluation should be repeated. This affects the transformer-based models since they are non-deterministic.
```

See how our model performs:
![alt text](img/fpfn_performance.png?raw=true)

The above figure shows analysis of performance vs. train budget, aggregated across datasets and prediction lengths. We plot the number of total MSE wins (left) where a higher value is better and mean MSE rank (right) where a lower values is better. Error bars show one standard deviation across training runs. ForecastPFN and Meta-N-BEATS are disadvantaged in these comparisons given that they see no training data for these series, only the length 36 input.

# Synthetic Data Generation <a name="SyntheticDataGeneration"></a>
ForecastPFN is completely trained on synthetic data.
To generate the synthetic data, use:

```
cd src/synthetic_generation
python main.py -c config.example.yaml
```

The config file has the following parameters which can be modified as per requirements:
- prefix : This denotes the parent directory where the synthetically generated data will be stored.
- version : Inside the prefix directory, a directory named [version name] will be created which will contain the daily, weekly and monthly tf records.
- sub_day : Setting this to True would create hourly and minutely records in addition to the normal records. In the paper, we have not used these sub day records for training ForecastPFN.
- num_series : This denotes how many series would be generated.

This is an example of generated synthetic data:
![alt text](img/synthetic_data_vis.png?raw=true)

# Model Training <a name="ModelTraining"></a>
We have trained ForecastPFN on a Tesla V100 16GB GPU, which takes around 30 hours.
To train the model, use:

```
cd src/training
python train.py -c config.example.yaml
```

The config file has the following parameters which can be modified as per requirements:
- prefix and version : These are the same as in synthetic data generation and denote which synthetic data the model should be trained on.
- model_save_name : The model will be saved in the prefix directory in the sub-directory models with this name.
- scaler : Set this to robust/max. We have discovered that robust scaling works better and provides faster convergence.
- test_noise : Setting this to True will add noise to the validation data as well. In our default setting, we only include noise in the training data.
- sub_day : Set this to True if the synthetic data contains sub-daily records.


# Citation 
Please cite our work if you use code from this repo:
```bibtex
@inproceedings{dooley2023forecastpfn,
  title={ForecastPFN: Synthetically-trained zero-shot forecasting},
  author={Dooley, Samuel and Khurana, Gurnoor Singh and Mohapatra, Chirag and Naidu, Siddartha V and White, Colin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
