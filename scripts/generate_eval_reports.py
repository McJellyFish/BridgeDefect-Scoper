#!/usr/bin/env python3
"""
Generate new LLM prediction reports for evaluation.

Author(s): [Author Name(s) - TO BE FILLED]
Date: [Date - TO BE FILLED]
Course: AIT 526 NLP - Natural Language Processing
Project: Multimodal Bridge Defect Scoper
Institution: George Mason University

- Case 1: images  -> YOLO + LLM (same logic as app_demo)
- Case 2: text    -> LLM-only with bridge keyword filter (same logic as app_demo)

Outputs (predictions):
  PROJECT_ROOT/evaluation_data/predictions/case1/reports/{i}.json
  PROJECT_ROOT/evaluation_data/predictions/case2/reports/{i:02d}.json
"""

import os
import sys
import json

from config import (
    CASE1_ORIG_IMAGES,
    CASE2_INPUTS_DIR,
    CASE1_LABELED_IMAGES,
    CASE1_PREDICTION_REPORTS,
    CASE2_PREDICTION_REPORTS,
    YOLO_CLASS_MAP
)
from yolo_model import run_yolo
from llmreport_demo import generate_bridge_report
from app_demo import parse_llm_report, BRIDGE_KEYWORDS

# ---------------------------------------------------------------------
# Paths - using centralized config
# ---------------------------------------------------------------------
# Convert Path objects to strings
CASE1_ORIG_IMAGES = str(CASE1_ORIG_IMAGES)
CASE2_INPUTS_DIR = str(CASE2_INPUTS_DIR)
CASE1_LABELLED_DIR = str(CASE1_LABELED_IMAGES)
CASE1_REPORTS_DIR = str(CASE1_PREDICTION_REPORTS)
CASE2_REPORTS_DIR = str(CASE2_PREDICTION_REPORTS)

# Create directories
os.makedirs(CASE1_LABELLED_DIR, exist_ok=True)
os.makedirs(CASE1_REPORTS_DIR, exist_ok=True)
os.makedirs(CASE2_REPORTS_DIR, exist_ok=True)

NON_BRIDGE_MESSAGE = "Please provide input related to bridge defects or inspection."


def _find_image(base_dir: str, stem: str):
    """
    Find image file with given stem name, trying multiple extensions.
    
    Helper function that searches for an image file when the exact extension
    is unknown. Tries common image formats to locate the file.
    
    Args:
        base_dir: Directory to search in
        stem: Base filename without extension (e.g., "1" for "1.jpg")
    
    Returns:
        Full path to image file if found, None otherwise.
    """
    # Try common image extensions in order
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    for ext in exts:
        path = os.path.join(base_dir, f"{stem}{ext}")
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------
# Case 1: Batch-generate reports for images (YOLO + LLM)
# ---------------------------------------------------------------------
def generate_case1_reports():
    """
    Batch-generate LLM reports for all image test cases.
    
    This function processes all images in the ground truth dataset by:
    1. Running YOLO detection on each image
    2. Extracting detected defects and confidence scores
    3. Generating LLM reports using Gemini API
    4. Saving reports as JSON files in predictions folder
    
    The workflow mirrors the main application (app_demo.py) to ensure
    consistency between evaluation and production use.
    
    Output:
        JSON report files saved to evaluation_data/predictions/case1/reports/
        Labeled images saved to evaluation_data/predictions/case1/labeled_images/
    
    Note:
        Skips files that already have generated reports to allow resumption
        of interrupted batch processing.
    """
    print("=== Generating Case 1 (image) reports into evaluation_data/predictions/ ===")

    # Dynamically scan for all image files in ground truth directory
    # This allows processing any number of images without hardcoding filenames
    image_files = []
    if os.path.exists(CASE1_ORIG_IMAGES):
        for filename in os.listdir(CASE1_ORIG_IMAGES):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                # Extract number from filename (e.g., "1.jpg" -> 1, "130.jpeg" -> 130)
                # Filenames should be numeric for consistent processing
                try:
                    img_num = int(os.path.splitext(filename)[0])
                    image_files.append((img_num, filename))
                except ValueError:
                    print(f"[warn] Skipping non-numeric image file: {filename}")
        
        # Sort by number to process in order
        image_files.sort(key=lambda x: x[0])
    else:
        print(f"[warn] Directory not found: {CASE1_ORIG_IMAGES}")
        return
    
    print(f"  Found {len(image_files)} image files to process")

    for img_num, filename in image_files:
        img_id = str(img_num)
        out_path = os.path.join(CASE1_REPORTS_DIR, f"{img_id}.json")

        if os.path.exists(out_path):
            print(f"[skip] case1_{img_id}: report already exists")
            continue

        image_path = os.path.join(CASE1_ORIG_IMAGES, filename)
        if not os.path.exists(image_path):
            print(f"[warn] case1_{img_id}: no image found at {image_path}")
            continue

        print(f"[info] case1_{img_id}: running YOLO on {filename}")

        # Run YOLO exactly like the app, but write labels to CASE1_LABELLED_DIR
        labelled_filename, yolo_results = run_yolo(image_path, CASE1_LABELLED_DIR)

        detected_labels = []
        confidence_data = {}

        # Default values if YOLO finds nothing
        label_summary = "No defects detected"
        image_for_llm = image_path

        if labelled_filename and yolo_results is not None and len(yolo_results.boxes) > 0:
            # This mirrors app_demo logic, but uses YOLO_CLASS_MAP from config
            label_summary = os.path.splitext(labelled_filename)[0]

            unique_classes = set(int(cls) for cls in yolo_results.boxes.cls)
            detected_labels = [
                YOLO_CLASS_MAP[cls_id]
                for cls_id in unique_classes
                if cls_id in YOLO_CLASS_MAP
            ]
            detected_labels = list(set(detected_labels))  # dedupe

            for idx, cls in enumerate(yolo_results.boxes.cls):
                class_id = int(cls)
                if class_id not in YOLO_CLASS_MAP:
                    continue
                class_name = YOLO_CLASS_MAP[class_id]
                conf = float(yolo_results.boxes.conf[idx])

                if class_name not in confidence_data:
                    confidence_data[class_name] = []
                confidence_data[class_name].append(conf)

            # For LLM context, send the labeled image with boxes (same idea as app)
            image_for_llm = os.path.join(CASE1_LABELLED_DIR, labelled_filename)

        # Call Gemini using the same interface as the app
        llm_text = generate_bridge_report(
            label_summary,
            image_path=image_for_llm,
            detected_labels=detected_labels,
        )

        # Use the same parser as the app (adds severity etc.)
        parsed_report = parse_llm_report(
            llm_text,
            detected_labels=detected_labels,
            confidence_data=confidence_data if confidence_data else None,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed_report, f, indent=2, ensure_ascii=False)

        print(f"[ok] case1_{img_id}: saved {out_path}")


# ---------------------------------------------------------------------
# Case 2: Batch-generate reports for text inputs (LLM-only)
# ---------------------------------------------------------------------


def generate_case2_reports():
    """
    Batch-generate LLM reports for all text input test cases.
    
    This function processes all text files in the ground truth dataset by:
    1. Reading text descriptions of bridge defects
    2. Validating bridge-related keywords (same logic as app_demo.py)
    3. Generating LLM reports using Gemini API
    4. Saving reports as JSON files in predictions folder
    
    For text inputs, there is no YOLO detection step - only LLM analysis.
    The reports focus on overall assessment and severity rating.
    
    Output:
        JSON report files saved to evaluation_data/predictions/case2/reports/
    
    Note:
        Skips files that already have generated reports to allow resumption
        of interrupted batch processing.
    """
    print("=== Generating Case 2 (text) reports into evaluation_data/predictions/ ===")

    # Dynamically scan for all text input files
    # Text files contain defect descriptions in natural language
    text_files = []
    if os.path.exists(CASE2_INPUTS_DIR):
        for filename in os.listdir(CASE2_INPUTS_DIR):
            if filename.lower().endswith('.txt'):
                # Extract number from filename (e.g., "01.txt" -> 1, "150.txt" -> 150)
                # Filenames should be numeric for consistent processing
                try:
                    text_num = int(os.path.splitext(filename)[0])
                    text_files.append((text_num, filename))
                except ValueError:
                    print(f"[warn] Skipping non-numeric text file: {filename}")
        
        # Sort by number to process in order
        text_files.sort(key=lambda x: x[0])
    else:
        print(f"[warn] Directory not found: {CASE2_INPUTS_DIR}")
        return
    
    print(f"  Found {len(text_files)} text input files to process")

    for text_num, filename in text_files:
        text_id = f"{text_num:02d}"
        in_path = os.path.join(CASE2_INPUTS_DIR, filename)
        out_path = os.path.join(CASE2_REPORTS_DIR, f"{text_id}.json")

        if not os.path.exists(in_path):
            print(f"[warn] case2_{text_id}: no input file {in_path}")
            continue

        if os.path.exists(out_path):
            print(f"[skip] case2_{text_id}: report already exists")
            continue

        with open(in_path, "r", encoding="utf-8") as f:
            text_input = f.read().strip()

        print(f"[info] case2_{text_id}: processing text (len={len(text_input)})")

        # EXACTLY like app_demo: keyword filter first
        lower = text_input.lower()
        matching_keywords = [kw for kw in BRIDGE_KEYWORDS if kw in lower]

        if not matching_keywords:
            # Not bridge-related → no LLM call, just a validation message
            parsed_report = {
                "valid": False,
                "message": NON_BRIDGE_MESSAGE,
                "original_input": text_input,
            }
        else:
            # Bridge-related text → call LLM
            llm_text = generate_bridge_report(
                text_input,
                image_path=None,
                detected_labels=None,
            )
            parsed_report = parse_llm_report(
                llm_text,
                detected_labels=[],
                confidence_data=None,
            )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed_report, f, indent=2, ensure_ascii=False)

        print(f"[ok] case2_{text_id}: saved {out_path}")


def main():
    """
    Main entry point for batch report generation.
    
    Orchestrates generation of prediction reports for both test case types:
    - Case 1: Image-based test cases (YOLO + LLM)
    - Case 2: Text-based test cases (LLM-only)
    
    Generated reports are saved to evaluation_data/predictions/ for subsequent
    evaluation against ground truth data.
    """
    generate_case1_reports()
    generate_case2_reports()
    from config import PREDICTIONS_DIR
    print("\nNew prediction reports written under:", str(PREDICTIONS_DIR))


if __name__ == "__main__":
    sys.exit(main())
