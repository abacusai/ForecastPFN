"""
Module to create train and test dfs
"""
import tensorflow as tf
import tensorflow_io
from prepare_dataset import gen_random_single_point, gen_mean_to_random_date, \
    gen_std_to_random_date, filter_unusable_points, build_frames, gen_random_single_point_no_noise, \
    gen_mean_to_random_date_no_noise, gen_std_to_random_date_no_noise


def remove_noise(x, y):
    """
    While training we do feed the noise into the model,
    so we remove it using this function
    :param x: contains the consolidated data including noise
    :param y: data point value for the given x
    """
    return (
        {
            'ts': x['ts'],
            'history': x['history'],
            'target_ts': x['target_ts'],
            'task': x['task']
        }, y
    )

def create_train_test_df(combined_ds, test_noise=False):
    """
    Method to create a train/test split from the combined_ds
    :param combined_ds: tf.Dataset containing data from all time frames
    such as daily, weekly and monthly
    :return: processed train test splits of the data
    """
    base_train_df = combined_ds.skip(30).map(build_frames).repeat()
    base_test_df = combined_ds.take(30).map(build_frames)
    task_map = {
        'point': gen_random_single_point,
        'mean': gen_mean_to_random_date,
        'stdev': gen_std_to_random_date
    }
    train_tasks_dfs = [
        base_train_df.map(func, num_parallel_calls=tf.data.AUTOTUNE)
        for func in task_map.values()
    ]
    train_df = tf.data.Dataset.choose_from_datasets(
        train_tasks_dfs, tf.data.Dataset.range(len(train_tasks_dfs)).repeat()
    ).unbatch().filter(filter_unusable_points)

    task_map_test = {
        'point': gen_random_single_point_no_noise,
        'mean': gen_mean_to_random_date_no_noise,
        'stdev': gen_std_to_random_date_no_noise
    }

    if test_noise:
        test_tasks_dfs = [
            base_test_df.map(func, num_parallel_calls=tf.data.AUTOTUNE)
            for func in task_map.values()
        ]
    else:
        test_tasks_dfs = [
            base_test_df.map(func, num_parallel_calls=tf.data.AUTOTUNE)
            for func in task_map_test.values()
        ]

    test_df = tf.data.Dataset.choose_from_datasets(
        test_tasks_dfs, tf.data.Dataset.range(len(test_tasks_dfs)).repeat()
    ).unbatch().filter(filter_unusable_points)

    # remove noise and target_noise from train and test df as they are now useless
    # train_df = train_df.map(remove_noise)
    test_df = test_df.map(remove_noise)

    return train_df, test_df
