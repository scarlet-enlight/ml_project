from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "Sleep_health_and_lifestyle_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "Sleep_health_and_lifestyle_dataset.csv",
    # ----------------------------------------------
):
    df = pd.read_csv(input_path)
    df['Gender'] = df['Gender'].map({'Male':0, 'Female': 1})

    unique_jobs = df['Occupation'].unique()
    df['Occupation'] = df['Occupation'].map({job:i for i, job in enumerate(unique_jobs)})

    # ---- Normal Weight is considered as Underweight, but csv file contained a meaning error, so we handled it like this below ----
    df.loc[df["BMI Category"] == "Normal Weight", "BMI Category"] = "Underweight"
    df['BMI Category'] = df['BMI Category'].map({'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3})

    sleep_disorders = df['Sleep Disorder'].unique()
    df['Sleep Disorder'] = df['Sleep Disorder'].map({diso:i for i, diso in enumerate(sleep_disorders)})

    cols = df['Blood Pressure'].str.split("/", expand=True)
    left_col = cols[0].astype(int)
    right_col = cols[1].astype(int)
    df.insert(9, "Blood Pressure Left", left_col)
    df.insert(10, "Blood Pressure Right", right_col)
    df = df.drop(columns=['Blood Pressure'])
    df.to_csv(output_path, index=0)


if __name__ == "__main__":
    app()
