from pathlib import Path
import pandas as pd
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "Sleep_health_and_lifestyle_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "Sleep_health_and_lifestyle_dataset.csv",
):
    df = pd.read_csv(input_path)
    print("Columns in original df:", df.columns.tolist())
    print("Unique values in 'Sleep Disorder':", df['Sleep Disorder'].unique())

    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    unique_jobs = df['Occupation'].unique()
    df['Occupation'] = df['Occupation'].map({job: i for i, job in enumerate(unique_jobs)})

    df.loc[df["BMI Category"] == "Normal Weight", "BMI Category"] = "Underweight"
    df['BMI Category'] = df['BMI Category'].map({'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3})

    sleep_disorders = df['Sleep Disorder'].unique()
    df['Sleep Disorder'] = df['Sleep Disorder'].map({diso: i for i, diso in enumerate(sleep_disorders)})
    print("Columns after mapping 'Sleep Disorder':", df.columns.tolist())
    print("Sample 'Sleep Disorder' values:", df['Sleep Disorder'].head())

    cols = df['Blood Pressure'].str.split("/", expand=True)
    left_col = pd.to_numeric(cols[0], errors='coerce').fillna(0).astype(int)
    right_col = pd.to_numeric(cols[1], errors='coerce').fillna(0).astype(int)

    df["Blood Pressure Left"] = left_col
    df["Blood Pressure Right"] = right_col
    df = df.drop(columns=['Blood Pressure'])

    df.to_csv(output_path, index=False)
    print("Columns in saved CSV:", pd.read_csv(output_path).columns.tolist())

if __name__ == "__main__":
    app()
