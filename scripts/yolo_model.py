#!/usr/bin/env python3
"""
YOLO Model Wrapper

Author(s): Sai Praneetha, Junjie Ma, I Chen Sung
Date: 12/03/2025
Course: AIT 526 NLP - Natural Language Processing
Project: Multimodal Bridge Defect Scoper
Institution: George Mason University

This module provides a wrapper for the YOLO object detection model.
It handles image processing and defect detection using a custom-trained
YOLO model specifically for bridge defect detection.
"""

import os
import torch
import logging
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")


# Ultralytics YOLO model.predict() usage:
# References:
#  https://docs.ultralytics.com/modes/predict/#inference-arguments

def run_yolo(image_path, labels_dir):
    """
    Run YOLO object detection on an image and save annotated result.
    
    This function uses a custom-trained YOLO model to detect bridge defects in images. 
    The model outputs bounding boxes around detected defects with confidence scores. 
    An annotated image is saved with bounding boxes and labels overlaid on the original image.
    
    Args:
        image_path: Path to the input image file to analyze
        labels_dir: Directory where the annotated output image will be saved
    
    Returns:
        Tuple of (labeled_filename, results_object):
        - labeled_filename: Name of the saved annotated image file (or None if failed)
        - results_object: YOLO results object containing detection data including:
          * boxes: Bounding box coordinates
          * cls: Class IDs of detected defects
          * conf: Confidence scores for each detection
    
    Note:
        The model uses the following configuration:
        - Image size: 640x640 pixels (YOLO standard)
        - Confidence threshold: 0.25 (detections below this are filtered)
        - IOU threshold: 0.7 (for non-maximum suppression)
        - Device: CPU (can be changed to 'cuda' if GPU available)
    
    Model Classes (5 defect types):
        0: bridge-crack
        1: for-review
        2: material-deterioration
        3: repair
        4: water-leakage
    """
    # Ensure output directory exists
    os.makedirs(labels_dir, exist_ok=True)
    
    # Load pre-trained YOLO model weights
    # Model file (best.pt) should be in the same directory as this script
    model = YOLO(MODEL_PATH)

    # Run YOLO prediction with specific configuration
    # The model.predict() method returns detection results and can save annotated images
    results = model.predict(
        source=image_path,        # Input image to process
        save=True,                # Save annotated image with bounding boxes
        project=labels_dir,       # Base folder for output
        name=".",                 # '.' forces direct save into project folder (no subfolder)
        exist_ok=True,            # Overwrite existing files if needed
        save_txt=False,           # Don't save .txt label files - we extract data from results object
        imgsz=640,                # Image size for inference (YOLO standard: 640x640)
        conf=0.25,                # Confidence threshold (0.0-1.0) - only detections above this are kept
        iou=0.7,                  # IOU threshold for Non-Maximum Suppression (removes overlapping boxes)
        device="cpu",             # Use CPU (change to "cuda" for GPU acceleration)
        show_conf=True,           # Display confidence scores on annotated image
        verbose=False             # Suppress YOLO progress output
    )

    # Find the latest saved annotated image in the labels directory
    # YOLO saves images with timestamps, so we find the most recent one
    images = [f for f in os.listdir(labels_dir) if f.lower().endswith(('.jpg', '.png'))]
    if images:
        # Get the most recently created image file
        latest = max(images, key=lambda x: os.path.getctime(os.path.join(labels_dir, x)))
        # Return filename and results object (results[0] is for first/only image)
        return os.path.basename(latest), results[0]
    return None, None
