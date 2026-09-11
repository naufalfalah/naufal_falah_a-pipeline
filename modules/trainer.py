"""run_fn untuk komponen Trainer (TFX) — model klasifikasi risiko penyakit jantung."""

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

from heart_disease_constants import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def _input_fn(file_pattern, tf_transform_output, batch_size=BATCH_SIZE):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


def _build_model(tf_transform_output):
    numerical_inputs = {
        transformed_name(key): tf.keras.Input(
            shape=(1,), name=transformed_name(key), dtype=tf.float32
        )
        for key in NUMERICAL_FEATURES
    }
    categorical_inputs = {
        transformed_name(key): tf.keras.Input(
            shape=(1,), name=transformed_name(key), dtype=tf.int64
        )
        for key in CATEGORICAL_FEATURES
    }

    embedded_categorical_features = []
    for key in CATEGORICAL_FEATURES:
        vocab_size = tf_transform_output.vocabulary_size_by_name(key)
        embed_dim = min(8, vocab_size)
        embedding = layers.Embedding(input_dim=vocab_size + 1, output_dim=embed_dim)(
            categorical_inputs[transformed_name(key)]
        )
        embedded_categorical_features.append(layers.Flatten()(embedding))

    concatenated = layers.concatenate(
        list(numerical_inputs.values()) + embedded_categorical_features
    )
    x = layers.Dense(64, activation="relu")(concatenated)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(
        inputs={**numerical_inputs, **categorical_inputs}, outputs=outputs
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, BATCH_SIZE)

    model = _build_model(tf_transform_output)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq="epoch"),
    ]

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
