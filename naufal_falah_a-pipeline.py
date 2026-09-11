"""Pipeline TFX: klasifikasi risiko penyakit jantung (Heart Disease UCI).

Dijalankan dengan Apache Beam (LocalDagRunner) sebagai Pipeline Orchestrator,
sesuai Kriteria 1 submission. Artifact tiap komponen (ExampleGen, StatisticsGen,
SchemaGen, ExampleValidator, Transform, Trainer, Evaluator, Pusher) disimpan
di folder `naufal_falah_a-pipeline/` (dibuat otomatis oleh TFX saat run).
"""

import os

# Harus di-set sebelum import TF pertama (lewat tfma), kalau tidak tf.keras terkunci ke Keras 3.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

PIPELINE_NAME = "naufal_falah_a-pipeline"

DATA_ROOT = "data_clean"
TRANSFORM_MODULE_FILE = os.path.join("modules", "transform.py")
TRAINER_MODULE_FILE = os.path.join("modules", "trainer.py")

PIPELINE_ROOT = PIPELINE_NAME
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata", "metadata.db")
SERVING_MODEL_DIR = os.path.join("serving_model", PIPELINE_NAME)


def init_components():
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    example_gen = CsvExampleGen(input_base=DATA_ROOT, output_config=output_config)

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=TRANSFORM_MODULE_FILE,
    )

    trainer = Trainer(
        module_file=TRAINER_MODULE_FILE,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=200),
    )

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("Latest_blessed_model_resolver")

    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(label_key="target", signature_name="serving_default")
        ],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=["sex"]),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(class_name="BinaryAccuracy"),
                    tfma.MetricConfig(class_name="Precision"),
                    tfma.MetricConfig(class_name="Recall"),
                    tfma.MetricConfig(
                        class_name="AUC",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.65}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-3},
                            ),
                        ),
                    ),
                ]
            )
        ],
    )

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=SERVING_MODEL_DIR
            )
        ),
    )

    return [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]


def run_pipeline():
    components = init_components()

    tfx_pipeline = pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
        beam_pipeline_args=[
            "--direct_running_mode=in_memory",
            "--direct_num_workers=1",
        ],
    )

    LocalDagRunner().run(tfx_pipeline)


if __name__ == "__main__":
    run_pipeline()
