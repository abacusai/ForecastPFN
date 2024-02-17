"""
Module to generate synthetic dataset for pre training
a time series forecasting model
"""
import argparse

import yaml
from config_variables import Config
from tf_generate_series import (
    convert_tf_to_rows,
    generate_product_input,
    load_tf_dataset,
    save_tf_records,
    tf_generate_n,
)


def save_tf_dataset(prefix: str, version: str, options: dict, num_series: int = 10_000):
    """
    Generate dataset and save as tf records
    """
    for freq, freq_index in Config.freq_and_index:
        print('Frequency: ' + freq)
        save_tf_records(
            prefix,
            f'{version}/{freq}.tfrecords',
            tf_generate_n(
                N=num_series,
                freq_index=freq_index,
                # start=pd.Timestamp("2020-01-01"),
                options=options,
            ),
        )


def generate_product_input_dataset(prefix, version):
    """
    Load dataset from tf records and save as avro files
    """
    for freq in Config.frequency_names:
        print('Frequency: ' + freq)
        generate_product_input(
            prefix,
            f'{version}/{freq}.avro',
            convert_tf_to_rows(
                load_tf_dataset(
                    prefix, f'{version}/{freq}.tfrecords'
                ).as_numpy_iterator()
            ),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to config file')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, yaml.loader.SafeLoader)

    Config.set_freq_variables(config['sub_day'])
    if 'transition' in config:
        Config.set_transition(config['transition'])

    save_tf_dataset(
        config['prefix'], config['version'], config['options'], config['num_series']
    )
    generate_product_input_dataset(config['prefix'], config['version'])


if __name__ == '__main__':
    main()
