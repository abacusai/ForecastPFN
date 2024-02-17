import tensorflow as tf

HISTORY_LEN = 100
MIN_CONTEXT = 20
PADDING = HISTORY_LEN - MIN_CONTEXT
TARGET_LEN = 10
TRIM_LEN = TARGET_LEN + max(0, MIN_CONTEXT - 2 * TARGET_LEN)
TARGET_INDEX = TARGET_LEN + TRIM_LEN

YEAR = 0
MONTH = 1
DAY = 2
DOW = 3
DOY = 4

# Tasks
SINGLE_POINT = 1
MEAN_TO_DATE = 2
STDEV_TO_DATE = 3
NUM_TASKS = 10

CONTEXT_LENGTH = 500
TF_SCHEMA = {
    "id": tf.io.FixedLenFeature([], dtype=tf.string),
    "ts": tf.io.FixedLenFeature([CONTEXT_LENGTH], dtype=tf.int64),
    "y": tf.io.FixedLenFeature([CONTEXT_LENGTH], dtype=tf.float32),
    "noise": tf.io.FixedLenFeature([CONTEXT_LENGTH], dtype=tf.float32),
}


# constant to reference where the academic_comparison and metalearning folders are
# will not be needed for training without validating on these datasets
ACADEMIC_HOME = "/home/ubuntu/ForecastPFN/academic_comparison/"
METALEARNED_HOME = ACADEMIC_HOME + "metalearned/"
