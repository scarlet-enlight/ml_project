from pathlib import Path

import pandas as pd
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from custom.knn import KNN
from custom.naive_bayes import NaiveBayes
from sklearn.model_selection import train_test_split


app = typer.Typer()

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_data(df: pd.DataFrame, target_column: str, train_ratio=0.6, random_state=42):
    # Mix the dataframe
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    split_index = int(len(df) * train_ratio)
    train_df = df[:split_index]
    test_df = df[split_index:]

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return X_train, X_test, y_train, y_test


@app.command()
def train(
    model: str = typer.Option(..., help="Type of model: knn, gnb"),
    data: str = typer.Option(..., help="Path to CSV with data"),
    target_column: str = typer.Option("target", help = "Label column"),
    train_ratio: float = typer.Option(0.6, help = "Percent of training data"),
    model_path: Path = MODELS_DIR / "model.pkl",
):
    df = load_dataset(data)
    X_train, X_test, y_train, y_test = split_data(df, target_column, train_ratio)

    if model == "knn":
        classifier = KNN(k=4)
    elif model == "gnb":
        classifier = NaiveBayes()
    else:
        typer.echo("Bad model name")
        raise typer.Exit(code=1)


    classifier.fit(X_train, y_train)

    counter = 0
    for row, target in zip(X_test.values, y_test.values):
        res = classifier.predict(row)
        if res == target:
            counter += 1
    acc = counter / len(X_test)
    typer.echo(f"Accuracy ({model.upper()}): {acc:.4f}")


if __name__ == "__main__":
    app()





