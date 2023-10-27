"""
Module to train the model
"""

from keras import backend
import yaml
import datetime
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_io
from utils import load_tf_dataset
from models import TransformerModel
from create_train_test_df import create_train_test_df
from config_variables import Config


def get_combined_ds(config):
    version = config["version"]

    # all the datasets we have. Ideally we use only 3 of these for trainig
    # adjust the values in this list accordingly
    datasets = [
        # load_tf_dataset(config["prefix"] + f"{version}/minute.tfrecords"),
        # load_tf_dataset(config["prefix"] + f"{version}/hourly.tfrecords"),
        load_tf_dataset(config["prefix"] + f"{version}/daily.tfrecords"),
        load_tf_dataset(config["prefix"] + f"{version}/weekly.tfrecords"),
        load_tf_dataset(config["prefix"] + f"{version}/monthly.tfrecords"),
    ]

    # # ucomment these lines to use the real world datasets in training
    # tourism_ds = load_tf_dataset(config['prefix'] + 'tourism.tfrecords')
    # wikiweb_ds = load_tf_dataset(config['prefix'] + 'wikiweb.tfrecords')

    combined_ds = tf.data.Dataset.choose_from_datasets(
        datasets, tf.data.Dataset.range(3).repeat()
    )

    return combined_ds


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, yaml.loader.SafeLoader)

    Config.set_sub_day(config["sub_day"])

    combined_ds = get_combined_ds(config)
    train_df, test_df = create_train_test_df(combined_ds, config["test_noise"])



    model = TransformerModel(scaler=config['scaler'])


    def smape(y_true, y_pred):
        """ Calculate Armstrong's original definition of sMAPE between `y_true` & `y_pred`.
        `loss = 200 * mean(abs((y_true - y_pred) / (y_true + y_pred), axis=-1)`
        Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        Returns:
        Symmetric mean absolute percentage error values. shape = `[batch_size, d0, ..
        dN-1]`.
        """
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        diff = tf.abs(
            (y_true - y_pred) / backend.maximum(y_true + y_pred, backend.epsilon())
        )
        return 200.0 * backend.mean(diff, axis=-1)

    # model = tf.keras.models.load_model(
    #     's3://realityengines.datasets/forecasting/pretrained/gurnoor/models/mf_replicate_testnoiseT_shuffle1Millilon.20230427-131143/ckpts/', custom_objects={'smape': smape}
    # )

    # need these two lines, else fit gives error
    batch_X, batch_y = next(iter(train_df.batch(2).take(1)))
    pred_y = model(batch_X)


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                 tf.keras.metrics.MeanSquaredError(name='mse'),
                 smape,
                 ]
    )


    fit_id = '.'.join([config["model_save_name"],
                       datetime.datetime.now().strftime("%Y%m%d-%H%M%S")])
    
    logdir = f"/home/ubuntu/tensorboard/notebook/pretrained/{fit_id}"
    tbCallback = tf.keras.callbacks.TensorBoard(logdir)
    tbCallback._val_dir = logdir+'/validation'
    callbacks = tf.keras.callbacks.CallbackList(
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                config["prefix"] + f"models/{fit_id}/ckpts", monitor="loss", verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                f"/home/ubuntu/tensorboard/notebook/pretrained/{fit_id}"
            ),
            # tf.keras.callbacks.LearningRateScheduler(
            #     lambda epoch, lr: min(0.001, lr * (epoch + 1))
            # )
            AdditionalValidationSets([(tourism_yearly_test_df, 'tourism_yearly'),
                                      (tourism_quarterly_test_df,'tourism_quarterly'),
                                      (tourism_monthly_test_df,'tourism_monthly'),
                                      (m3_yearly_test_df, 'm3_yearly'),
                                      (m3_quarterly_test_df, 'm3_quarterly'),
                                      (m3_monthly_test_df, 'm3_monthly'),
                                      (m3_others_test_df, 'm3_others'),
                                      ], 
                                      tbCallback)
        ],
        add_history=True,
        add_progbar=True,
        model=model,
    )


    model.fit(
        train_df.shuffle(5_000_000, reshuffle_each_iteration=True).batch(
            1024).prefetch(tf.data.AUTOTUNE),
        # train_df.take(1000_000).cache().shuffle(100_000).batch(1024).prefetch(tf.data.AUTOTUNE),
        validation_data=test_df.batch(1024, drop_remainder=False).cache(),
        epochs=700,
        steps_per_epoch=1000,
        callbacks=callbacks,
    )

    model.save(config["prefix"] + 'models/'+ config["model_save_name"])


if __name__ == "__main__":
    main()
