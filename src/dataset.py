from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "tested.csv",
    output_path: Path = PROCESSED_DATA_DIR / "tested.csv",
):
    df = pd.read_csv(input_path)
    df = df.drop(columns = ['Ticket', 'Name','Cabin', 'Embarked'])
    df['Sex'] = df['Sex'].map({'female':0, 'male': 1})
    df.fillna(df.mean(), inplace=True)
    df.to_csv(output_path, index=False)
    df.describe()
    df.info()
    



if __name__ == "__main__":
    app()
