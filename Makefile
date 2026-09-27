# ============================================================
# Bearing Fault Diagnosis Experiment - Makefile
# Usage:
#   make install      Install dependencies
#   make preprocess   Raw time series -> features (tsfel)
#   make features     Features -> PCA dimensionality reduction
#   make train        Train/evaluate classifiers
#   make all          Full pipeline
#   make clean        Clean intermediate artifacts
# ============================================================

PYTHON      ?= python
PIP         ?= pip
DATA_DIR    := ./data
RAW_DIR     := $(DATA_DIR)/江南大学轴承数据(一)/数据

# Raw time series data
IB_RAW      := $(RAW_DIR)/ib600_2.csv
OB_RAW      := $(RAW_DIR)/ob600_2.csv
TB_RAW      := $(RAW_DIR)/tb600_2.csv
NS_RAW      := $(RAW_DIR)/n600_3_2.csv

# Feature files
IB_FEAT     := $(DATA_DIR)/ib600_2_features.csv
OB_FEAT     := $(DATA_DIR)/ob600_2_features.csv
TB_FEAT     := $(DATA_DIR)/tb600_2_features.csv
NS_FEAT     := $(DATA_DIR)/n600_3_2_features.csv

.PHONY: help install spacy-model preprocess features train all clean

help:
	@echo "Available targets:"
	@echo "  make install      Install dependencies from requirements.txt"
	@echo "  make spacy-model  Download spacy (en_core_web_sm)"
	@echo "  make preprocess   Raw time series -> features (tsfel)"
	@echo "  make features     Features -> PCA dimensionality reduction"
	@echo "  make train        Train/evaluate classifiers (Naive Bayes + Chaos ensemble)"
	@echo "  make all          Full pipeline (preprocess -> features -> train)"
	@echo "  make clean        Clean intermediate feature files"

# ---- Environment ----
install:
	$(PIP) install -r requirements.txt

spacy-model:
	$(PYTHON) -m spacy download en_core_web_sm

# ---- Data pipeline ----
preprocess: $(IB_FEAT) $(OB_FEAT) $(TB_FEAT) $(NS_FEAT)

$(IB_FEAT): $(IB_RAW) pre_process_data.py
	$(PYTHON) pre_process_data.py

$(OB_FEAT): $(OB_RAW) pre_process_data.py
	$(PYTHON) pre_process_data.py

$(TB_FEAT): $(TB_RAW) pre_process_data.py
	$(PYTHON) pre_process_data.py

$(NS_FEAT): $(NS_RAW) pre_process_data.py
	$(PYTHON) pre_process_data.py

features:
	$(PYTHON) decomposition_features.py

train:
	$(PYTHON) main_merge_classfier.py

compare:
	$(PYTHON) compare.py

# ---- Combined ----
all: preprocess features train

clean:
	rm -f $(IB_FEAT) $(OB_FEAT) $(TB_FEAT) $(NS_FEAT)
	rm -f $(DATA_DIR)/*_pca*.csv
	rm -f $(DATA_DIR)/*_pca*.pkl
	rm -f $(DATA_DIR)/*_scaler*.pkl
	@echo "Cleaned intermediate feature files"
