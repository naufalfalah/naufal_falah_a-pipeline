"""preprocessing_fn untuk komponen Transform (TFX)."""

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_transform as tft

from heart_disease_constants import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)


def preprocessing_fn(inputs):
    """Feature engineering: scaling untuk fitur numerik, vocabulary untuk kategorikal."""
    outputs = {}

    for key in NUMERICAL_FEATURES:
        outputs[transformed_name(key)] = tft.scale_to_z_score(
            tf.cast(inputs[key], tf.float32)
        )

    for key in CATEGORICAL_FEATURES:
        outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(
            inputs[key], vocab_filename=key, num_oov_buckets=1
        )

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
