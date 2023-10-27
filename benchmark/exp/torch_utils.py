import torch
import os
from collections import OrderedDict
from functools import partial
from torch.nn import MSELoss
from torch.optim import Adam
import tensorflow_datasets as tfds
import tensorflow as tf
from tqdm import tqdm

DEFAULT_LOSS = MSELoss()
DEFAULT_OPTIMIZER = partial(Adam, lr=0.001)


load_dir = 'tensorboard/mf_replicate_testnoiseT_shuffle5Millilon.20230714-133237/models/51'


def numpy_to_torch(X, device):
    if type(X) == dict:
        X_torch = {}
        for k, v in X.items():
            X_torch[k] = torch.tensor(v, dtype=torch.float64).to(device)
    else:
        X_torch = torch.from_numpy(X).to(device)
    return X_torch


def get_dataset(ds, batch_size=16, shuffle=0):
    ds = ds.cache()
    if shuffle > 0:
        ds = ds.shuffle(shuffle)
        # opt = tf.data.Options()
        # opt.experimental_deterministic = False
        # ds = ds.with_options(opt)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(ds)


class TFRecordDataLoader:
    def __init__(self, ds, batch_size=1024, train=True, shuffle=0):
        self.ds = get_dataset(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.batch_size = batch_size
        self._iterator = None

    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self.ds)
        else:
            self._reset()
        return self._iterator

    def _reset(self):
        self._iterator = iter(self.ds)

    def __next__(self):
        batch = next(self._iterator)
        return batch


class AdditionalValidationSets:
    def __init__(self, validation_sets, batch_size=1, metrics=[], loss=DEFAULT_LOSS, device=None):
        self.validation_sets = []
        for validation_set in validation_sets:
            if len(validation_set) not in [2]:
                raise ValueError()
            self.validation_sets.append([tfds.as_numpy(validation_set[0]), validation_set[1]])
        self.epoch = []
        self.metrics = metrics
        self.loss = loss
        self.device = device or torch.device('cpu')
        self.logs = {}

    def on_train_begin(self):
        self.epoch = []
        self.logs = {}

    def on_epoch_end(self, model, epoch, tbCallback=None):
        log = {}
        self.epoch.append(epoch)
        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 2:
                validation_data, validation_set_name = validation_set
            else:
                raise ValueError()
            results = add_metrics_to_log(model, validation_data, self.loss, self.metrics, tbCallback, f'add_valid/{validation_set_name}/', epoch, self.device)
            log.update(results)
        self.logs[epoch] = log
        return log


def predict(model, data, device, steps_per_epoch=None):
    # Batch prediction
    model.eval()
    y_pred = None
    y_true = None
    for batch_i, batch_data in enumerate(data):
        # Predict on batch
        X_batch = numpy_to_torch(batch_data[0], device)
        y_batch = torch.from_numpy(batch_data[1]).to(device)
        y_batch_pred = model(X_batch)
        y_batch_pred, y_batch = model.transform_output(y_batch_pred, y_batch)
        y_true = y_batch if y_true is None else torch.concat([y_true, y_batch])
        y_pred = y_batch_pred if y_pred is None else torch.concat([y_pred, y_batch_pred])
        if steps_per_epoch is not None and batch_i >= steps_per_epoch:
            break
    return y_true, y_pred


def add_metrics_to_log(model, data, loss, metrics, writer, prefix, epoch, device, steps_per_epoch=None):
    with torch.no_grad():
        y_true, y_pred = predict(model, data, device, steps_per_epoch)
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
    log = {}
    for metric in metrics:
        q = metric(y_true, y_pred).item()
        log[prefix + metric.__name__] = q
        if writer is not None:
            writer.add_scalar(prefix + metric.__name__, q, epoch)
    loss_value = loss(y_pred, y_true).item()
    log[prefix + 'loss'] = loss_value
    if writer is not None:
        writer.add_scalar(prefix + 'loss', loss_value, epoch)
    return log


def fit(model,
        train_df,
        batch_size=1024,
        epochs=1,
        verbose=1,
        valid_df=None,
        shuffle=0,
        initial_epoch=0,
        seed=None,
        loss=DEFAULT_LOSS,
        optimizer=DEFAULT_OPTIMIZER,
        metrics=None,
        writer=None,
        device='cpu',
        steps_per_epoch=None,
        logdir=None,
        additional_validation_sets=[]):
    """Trains the model similar to Keras' .fit(...) method

    # Arguments
        train_df: Generator for train data
        batch_size: integer. Number of samples per gradient update.
        epochs: integer, the number of times to iterate
            over the training data arrays.
        verbose: 0, 1. Verbosity mode.
            0 = silent, 1 = verbose.
        valid_df: Generator for validation data on which to evaluate
            the loss and any model metrics
            at the end of each epoch. The model will not
            be trained on this data.
        shuffle: boolean, whether to shuffle the training data
            before each epoch.
        initial_epoch: epoch at which to start training
            (useful for resuming a previous training run)
        seed: random seed.
        optimizer: training optimizer
        loss: training loss
        metrics: list of functions with signatures `metric(y_true, y_pred)`
            where y_true and y_pred are both Tensors
        writer: tensorboard summary writer
    # Returns
        list of OrderedDicts with training metrics
    """
    if seed and seed >= 0:
        torch.manual_seed(seed)

    logdir = logdir or '.'
    os.makedirs(logdir + '/models/', exist_ok=True)
    logfile = open(logdir + '/logs', 'a')
    logfile.write('-' * 32 + '\n')

    # Build DataLoaders
    valid_data = TFRecordDataLoader(valid_df, batch_size)
    additional_valid_data = AdditionalValidationSets(additional_validation_sets, metrics=metrics, loss=loss, device=device)
    # Compile optimizer
    opt = optimizer(model.parameters())
    # load = torch.load(load_dir)
    # model.load_state_dict(load['model'])
    # opt.load_state_dict(load['optimizer'])
    # Run training loop
    logs = []
    for t in tqdm(range(initial_epoch, epochs)):
        logfile.write(f"Epoch: {t+1}\n")
        train_data = TFRecordDataLoader(train_df, batch_size, True, shuffle)
        model.train()
        if verbose and t % 10 == 0:
            print("Epoch {0} / {1}".format(t + 1, epochs))
        log = OrderedDict()
        epoch_loss = 0.0
        # Run batches
        for batch_i, batch_data in enumerate(train_data):
            # Get batch data
            X_batch = numpy_to_torch(batch_data[0], device)
            y_batch = torch.from_numpy(batch_data[1]).to(device)
            # Backprop
            opt.zero_grad()
            y_batch_pred = model(X_batch)
            y_batch_pred, y_batch = model.transform_output(y_batch_pred, y_batch)
            batch_loss = loss(y_batch_pred, y_batch)
            batch_loss.backward()
            opt.step()
            # Update status
            epoch_loss += batch_loss.item()
            log['loss'] = float(epoch_loss) / (batch_i + 1)
            if steps_per_epoch is not None and batch_i >= steps_per_epoch:
                break
        if writer is not None:
            writer.add_scalar('train/epoch/loss', log['loss'], t)
        # Run metrics
        # train_metric_log = add_metrics_to_log(model, train_data, loss, metrics, writer, prefix='train/metrics/', epoch=t, device=device, steps_per_epoch=steps_per_epoch)
        # log.update(train_metric_log)
        if valid_data is not None:
            val_metric_log = add_metrics_to_log(model, valid_data, loss, metrics, writer, prefix='valid/metrics/', epoch=t, device=device)
            log.update(val_metric_log)
        # Additional validation set
        if t % 10 == 0:
            add_log = additional_valid_data.on_epoch_end(model, t, writer)
            logfile.write(str(add_log)+'\n')
            to_save = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
            }
            torch.save(to_save, logdir + f'/models/{t+1}')
        logfile.write(str(log)+'\n')
        logfile.flush()
        logs.append(log)

    logfile.close()
    return logs, model, opt, additional_valid_data
