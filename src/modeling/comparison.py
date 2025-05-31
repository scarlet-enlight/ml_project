from pathlib import Path
import typer
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from custom.knn import KNN
from custom.naive_bayes import NaiveBayes
from train import load_dataset, split_data
from src.config import PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def compare_models(
    data_path: Path = typer.Option(PROCESSED_DATA_DIR / "Sleep_health_and_lifestyle_dataset.csv", help="Path to CSV data"),
    target_column: str = typer.Option("Sleep Disorder", help="Target column name"),
    train_ratio: float = typer.Option(0.6, help="Train-test split ratio")
):
    df = load_dataset(str(data_path))

    df.columns = df.columns.str.strip().str.lower()
    target_column_normalized = target_column.strip().lower()

    if target_column_normalized not in df.columns:
        typer.secho(f"Column '{target_column}' not found in dataset!", fg=typer.colors.RED)
        typer.echo(f"Available columns: {df.columns.tolist()}")
        raise typer.Exit(code=1)
    
    X_train, X_test, y_train, y_test = split_data(df, target_column_normalized, train_ratio)

    models = {
        "KNN": KNN(k=4),
        "Naive Bayes": NaiveBayes()
    }

    for name, model in models.items():
        typer.echo(f"\n========== {name.upper()} ==========")
        model.fit(X_train, y_train)

        y_pred = [model.predict(x) for x in X_test.values]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        typer.echo(f"Accuracy       : {acc:.4f}")
        typer.echo(f"Precision (avg): {prec:.4f}")
        typer.echo(f"Recall (avg)   : {rec:.4f}")
        typer.echo(f"F1-score (avg) : {f1:.4f}")
        typer.echo("\nConfusion Matrix:")
        typer.echo(pd.DataFrame(cm))
        typer.echo("\nClassification Report:")
        typer.echo(report)


if __name__ == "__main__":
    app()
