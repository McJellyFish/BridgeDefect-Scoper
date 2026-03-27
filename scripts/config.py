#!/usr/bin/env python3
"""
Evaluation Dataset Path Configuration

Author(s): Sai Praneetha, Junjie Ma, I Chen Sung
Date: 12/03/2025
Course: AIT 526 NLP - Natural Language Processing
Project: Multimodal Bridge Defect Scoper
Institution: George Mason University

This module provides centralized path configuration for all evaluation scripts.
Users can override paths using environment variables if dataset is in a different location.
The folder structure was reorganized from "evaluation/" and "evaluation_new/" to
"evaluation_data/ground_truth/" and "evaluation_data/predictions/" for clarity.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

# ---------------------------------------------------------------------
# Evaluation Dataset Paths
# ---------------------------------------------------------------------
# Default paths (relative to project root)
# NOTE: Structure changed from "evaluation/" and "evaluation_new/" 
#       to "evaluation_data/ground_truth/" and "evaluation_data/predictions/"
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "evaluation_data"
DEFAULT_GROUND_TRUTH_DIR = DEFAULT_EVAL_ROOT / "ground_truth"
DEFAULT_PREDICTIONS_DIR = DEFAULT_EVAL_ROOT / "predictions"

# Allow override via environment variables
EVAL_ROOT = Path(os.getenv("EVAL_ROOT_DIR", str(DEFAULT_EVAL_ROOT)))
GROUND_TRUTH_DIR = Path(os.getenv("EVAL_GROUND_TRUTH_DIR", str(DEFAULT_GROUND_TRUTH_DIR)))
PREDICTIONS_DIR = Path(os.getenv("EVAL_PREDICTIONS_DIR", str(DEFAULT_PREDICTIONS_DIR)))

# ---------------------------------------------------------------------
# Specific Paths
# ---------------------------------------------------------------------
# Ground Truth Paths - Reference data for evaluation
GROUND_TRUTH_CSV = GROUND_TRUTH_DIR / "ground_truth.csv"  # Metadata CSV with test case info
CASE1_GROUND_TRUTH = GROUND_TRUTH_DIR / "case1"           # Image test cases directory
CASE2_GROUND_TRUTH = GROUND_TRUTH_DIR / "case2"           # Text test cases directory
CASE1_ORIG_IMAGES = CASE1_GROUND_TRUTH / "original_images"  # Original test images
CASE1_LABELS = CASE1_GROUND_TRUTH / "labels"              # YOLO detection labels (.txt files)
CASE2_INPUTS_DIR = CASE2_GROUND_TRUTH / "inputs"          # Text input files (.txt)

# Prediction Paths - Generated reports from system
CASE1_PREDICTIONS = PREDICTIONS_DIR / "case1"             # Image case predictions directory
CASE2_PREDICTIONS = PREDICTIONS_DIR / "case2"             # Text case predictions directory
CASE1_PREDICTION_REPORTS = CASE1_PREDICTIONS / "reports"  # LLM-generated JSON reports for images
CASE2_PREDICTION_REPORTS = CASE2_PREDICTIONS / "reports"  # LLM-generated JSON reports for text
CASE1_LABELED_IMAGES = CASE1_PREDICTIONS / "labeled_images"  # YOLO-labeled images with bounding boxes

# Results Output Paths - Evaluation metrics and visualizations
RESULTS_DIR = GROUND_TRUTH_DIR / "results"                # Base directory for all evaluation outputs
PLOTS_DIR = RESULTS_DIR / "plots"                         # Objective evaluation visualization plots
DEEPEVAL_PLOTS_DIR = RESULTS_DIR / "deepeval_plots"       # DeepEval metric visualization plots
EVALUATION_RESULTS_CSV = RESULTS_DIR / "evaluation_results.csv"  # Per-case detailed results
EVALUATION_SUMMARY_TXT = RESULTS_DIR / "evaluation_summary.txt"  # Aggregate statistics
DEEPEVAL_RESULTS_CSV = RESULTS_DIR / "deepeval_results.csv"  # DeepEval metric scores
DEEPEVAL_SUMMARY_TXT = RESULTS_DIR / "deepeval_summary.txt"  # DeepEval aggregate stats

# ---------------------------------------------------------------------
# YOLO Class Mapping
# ---------------------------------------------------------------------
YOLO_CLASS_MAP = {
    0: "bridge-crack",
    1: "for-review",
    2: "material-deterioration",
    3: "repair",
    4: "water-leakage"
}

