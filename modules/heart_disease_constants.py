"""Definisi fitur untuk pipeline klasifikasi risiko penyakit jantung.

Dipakai bersama oleh transform.py dan trainer.py supaya nama-nama fitur
konsisten di seluruh komponen (Transform, Trainer, Evaluator).
"""

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "ca",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
]

LABEL_KEY = "target"


def transformed_name(key: str) -> str:
    """Menambahkan suffix `_xf` untuk membedakan fitur hasil Transform."""
    return f"{key}_xf"
