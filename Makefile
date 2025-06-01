#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ml_project
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) src/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)


#################################################################################
# Machine learning commands                                                     #
#################################################################################

# Default parameters
DATA=data/processed/Sleep_health_and_lifestyle_dataset.csv
TARGET='Sleep Disorder'
# PREDICT_DATA
# OUTPUT_KNN=predictions.csv

## Training a KNN classificator
.PHONY: train-knn
train-knn:
	@echo "Training a KNN model"
	@$(PYTHON_INTERPRETER) src/modeling/train.py --model knn --data $(DATA) --target-column $(TARGET)

## Training a Naive Bayes classificator
.PHONY: train-gnb
train-gnb:
	@echo "Training a Gaussian Naive Bayes model"
	@$(PYTHON_INTERPRETER) src/modeling/train.py --model gnb --data $(DATA) --target-column $(TARGET)

## Training a Decision Tree classificator
.PHONY: train-tre
train-tre:
	@echo "Training a Decision Tree model"
	@$(PYTHON_INTERPRETER) src/modeling/train.py --model tre --data $(DATA) --target-column $(TARGET)

## Training all models
.PHONY: train-all
train-all: train-knn train-gnb train-tre
	@echo "Models were trained"

## Launching full program
.PHONY: launch
launch:
	@echo "Starting a Model Comparison."
	python src/modeling/comparison.py