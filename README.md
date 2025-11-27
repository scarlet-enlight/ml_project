# Sleep Health ML Project

A concise student project comparing K-Nearest Neighbors, Gaussian Naive Bayes, and Decision Tree classifiers on the "Sleep Health and Lifestyle" dataset. The notebooks show data exploration, preprocessing, model training, and evaluation.

## Quick summary

- **Purpose:** Educational comparison of simple classifiers and basic ML workflow.
- **Models compared:** KNN, Gaussian Naive Bayes, Decision Tree.
- **Metrics:** Accuracy, precision, recall, F1-score, confusion matrix.

## Dataset

The project uses the Sleep Health and Lifestyle dataset (source included in the repository under `data/raw`). If the dataset is external, include the original link or instructions to download it into `data/raw`.

## How to run

1. Install requirements:  
   `pip install -r requirements.txt`
2. Open the Jupyter notebooks in the `notebooks/` directory (they are ordered by steps: data exploration → preprocessing → modeling → evaluation).
3. Run cells step-by-step. Notebooks save model artifacts to `models/` and figures to `reports/figures/`.

For a reproducible run with scripts (if available):
- `python src/modeling/train.py`  # trains models
- `python src/modeling/predict.py`  # runs inference on test data

## Results

See `reports/` for evaluation tables and plots. The notebooks include a concise comparison table and confusion matrices for each classifier.

## Repository structure

- `data/` — datasets (raw, processed)
- `notebooks/` — Jupyter Notebooks with all steps
- `src/` — source code (training, inference, preprocessing)
- `models/` — trained models
- `reports/` — results, figures, tables
- `docs/` — documentation (mkdocs)

## Contributing

This is a student project. Contributions are welcome but please keep changes small and documented.

## License & Contact

MIT License (if included). For questions, contact the project owner.

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
