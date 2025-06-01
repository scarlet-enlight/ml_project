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

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = "data/"
PROCESSED_DATA_DIR = f'{DATA_DIR}processed/'
from custom.knn import KNN
from custom.naive_bayes import NaiveBayes
from custom.decision_tree import DecisionTree
from train import load_dataset, split_data
from src.config import PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def compare_models(
    data_path: Path = typer.Option(f"{PROCESSED_DATA_DIR}Sleep_health_and_lifestyle_dataset.csv", help="Path to CSV data"),
    target_column: str = typer.Option("Sleep Disorder", help="Target column name"),
    train_ratio: float = typer.Option(0.6, help="Train-test split ratio"),
    show_details: bool = typer.Option(False, help="Show confusion matrix and classification report")
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
        "Naive Bayes": NaiveBayes(),
        "Decision Tree": DecisionTree()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = [model.predict(x) for x in X_test.values]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })

        if show_details:
            typer.echo(f"\n----- {name} -----")
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)
            typer.echo("Confusion Matrix:")
            typer.echo(pd.DataFrame(cm))
            typer.echo("\nClassification Report:")
            typer.echo(report)

    df_results = pd.DataFrame(results).set_index("Model")
    typer.echo("\n=== Model Comparison Summary ===")
    typer.echo(df_results.round(4))

    html_output = Path(__file__).parent / "model_comparison.html"
    styled = df_results.style \
        .highlight_max(color="lightgreen", axis=0) \
        .set_caption("Model Performance Comparison") \
        .format("{:.4f}") \
        .set_table_styles([
            {"selector": "caption", "props": [("caption-side", "top"), ("font-size", "1.2em"), ("font-weight", "bold")]},
            {"selector": "th", "props": [("background-color", "#f2f2f2"), ("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]}
        ])
    styled.to_html(html_output)

    typer.secho(f"\nComparison HTML saved to: {html_output}", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()
