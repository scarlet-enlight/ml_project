from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from models.knn_model import knn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "tested.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    #kod knn
    df = pd.read_csv(input_path)
    X = df.drop(columns = 'Survived')
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.4)
    print(X_train)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    



if __name__ == "__main__":
    app()
