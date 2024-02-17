"""
Module to train the model
"""

import argparse
import datetime

import numpy as np
import tensorflow as tf
import yaml
from config_variables import Config
from create_train_test_df import create_train_test_df
from keras import backend
from models import TransformerModel
from utils import load_tf_dataset


def get_combined_ds(config):
    version = config["version"]

    # all the datasets we have. Ideally we use only 3 of these for trainig
    # adjust the values in this list accordingly
    datasets = [
        load_tf_dataset(config["prefix"] + f"{version}/daily.tfrecords"),
        load_tf_dataset(config["prefix"] + f"{version}/weekly.tfrecords"),
        load_tf_dataset(config["prefix"] + f"{version}/monthly.tfrecords"),
    ]

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

    model = TransformerModel(scaler=config["scaler"])

    def smape(y_true, y_pred):
        """Calculate Armstrong's original definition of sMAPE between `y_true` & `y_pred`.
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

    # need these two lines, else fit gives error
    batch_X, batch_y = next(iter(train_df.batch(2).take(1)))
    model(batch_X)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
            smape,
        ],
    )

    class AdditionalValidationSets(tf.keras.callbacks.Callback):
        def __init__(self, validation_sets, tbCallback, verbose=1, batch_size=1):
            """
            :param validation_sets:
            a list of 2-tuples (validation_data, validation_set_name)
            or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
            :param verbose:
            verbosity mode, 1 or 0
            :param batch_size:
            batch size to be used when evaluating on the additional datasets
            """
            super(AdditionalValidationSets, self).__init__()
            self.validation_sets = validation_sets
            for validation_set in self.validation_sets:
                if len(validation_set) not in [2]:
                    raise ValueError()
            self.epoch = []
            self.tbCallback = tbCallback
            self.history = {}
            self.verbose = verbose
            self.batch_size = batch_size

        def on_train_begin(self, logs=None):
            self.epoch = []
            self.history = {}

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.epoch.append(epoch)

            # record the same values as History() as well
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

            # evaluate on the additional validation sets
            for validation_set in self.validation_sets:
                if len(validation_set) == 2:
                    validation_data, validation_set_name = validation_set
                    sample_weights = None
                else:
                    raise ValueError()

                print(validation_set_name)
                results = self.model.evaluate(
                    x=validation_data,
                    verbose=self.verbose,
                    sample_weight=sample_weights,
                    batch_size=self.batch_size,
                )

                for metric, result in zip(self.model.metrics_names, results):
                    valuename = validation_set_name + "_" + metric
                    self.history.setdefault(valuename, []).append(result)
                    with self.tbCallback._val_writer.as_default(step=epoch):
                        tf.summary.scalar(valuename, result)

    fit_id = ".".join(
        [config["model_save_name"], datetime.datetime.now().strftime("%Y%m%d-%H%M%S")]
    )

    logdir = f"/home/ubuntu/tensorboard/notebook/pretrained/{fit_id}"
    tbCallback = tf.keras.callbacks.TensorBoard(logdir)
    tbCallback._val_dir = logdir + "/validation"
    callbacks = tf.keras.callbacks.CallbackList(
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                config["prefix"] + f"models/{fit_id}/ckpts", monitor="loss", verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                f"/home/ubuntu/tensorboard/notebook/pretrained/{fit_id}"
            ),
        ],
        add_history=True,
        add_progbar=True,
        model=model,
    )

    model.fit(
        train_df.shuffle(5_000, reshuffle_each_iteration=True)
        .batch(1024)
        .prefetch(tf.data.AUTOTUNE),
        # train_df.take(1000_000).cache().shuffle(100_000).batch(1024).prefetch(tf.data.AUTOTUNE),
        validation_data=test_df.batch(1024, drop_remainder=False).cache(),
        epochs=700,
        steps_per_epoch=10,
        callbacks=callbacks,
    )

    model.save(config["prefix"] + "models/" + config["model_save_name"])


if __name__ == "__main__":
    main()
