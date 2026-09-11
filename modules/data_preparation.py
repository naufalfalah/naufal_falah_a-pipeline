"""Membersihkan dataset mentah Heart Disease (UCI) sebelum masuk ke CsvExampleGen.

TFX ExampleGen tidak melakukan imputasi/cleaning, jadi tahap ini dijalankan
sekali di luar graph pipeline untuk menghasilkan CSV yang sudah rapi:

- Kolom `id` dan `dataset` (asal rumah sakit) dibuang bukan fitur/berpotensi leakage.
- Label `num` (0-4, tingkat keparahan) dibinerkan jadi `target` (0 = sehat, 1 = berisiko).
- Fitur numerik yang kosong diisi median kolom tersebut.
- Fitur kategorikal yang kosong diisi kategori baru "missing" (missingness-nya
  sendiri berpotensi informatif, bukan dibuang).

Jalankan sekali: `python modules/data_preparation.py`
"""

import os

import pandas as pd

from heart_disease_constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, LABEL_KEY

RAW_CSV_PATH = os.path.join("data", "heart_disease_uci.csv")
CLEAN_CSV_DIR = "data_clean"
CLEAN_CSV_PATH = os.path.join(CLEAN_CSV_DIR, "heart_disease_clean.csv")

COLUMNS_TO_DROP = ["id", "dataset"]
RAW_LABEL_COLUMN = "num"


def clean_dataset(raw_csv_path: str = RAW_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)

    df = df.drop(columns=COLUMNS_TO_DROP)

    df[LABEL_KEY] = (df[RAW_LABEL_COLUMN] > 0).astype(int)
    df = df.drop(columns=[RAW_LABEL_COLUMN])

    for column in NUMERICAL_FEATURES:
        df[column] = df[column].fillna(df[column].median())

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("missing").astype(str)

    ordered_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [LABEL_KEY]
    return df[ordered_columns]


def main():
    df = clean_dataset()
    os.makedirs(CLEAN_CSV_DIR, exist_ok=True)
    df.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"Tersimpan {len(df)} baris ke {CLEAN_CSV_PATH}")
    print(f"Distribusi target:\n{df[LABEL_KEY].value_counts()}")
    print(f"Sisa missing value:\n{df.isna().sum()[df.isna().sum() > 0]}")


if __name__ == "__main__":
    main()
