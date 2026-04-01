#!/usr/bin/env python3
"""
Main Flask Web Application

Author(s): Sai Praneetha, Junjie Ma, I Chen Sung
Date: 12/03/2025
Course: AIT 526 NLP - Natural Language Processing
Project: Multimodal Bridge Defect Scoper
Institution: George Mason University

This is the main Flask web application for the Bridge Defect Scoper system.
It handles image and text input, coordinates YOLO detection and LLM report generation, and displays results through a web interface.
"""

import re
import json
from yolo_model import run_yolo
from llmreport_demo import generate_bridge_report, DEFECT_KNOWLEDGE_BASE
from dotenv import load_dotenv
from ultralytics import YOLO
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
import os
import secrets
import json
import sys
import shutil
import markdown
import logging
import warnings


# ---------------------------------------------------------------------
# Suppress warnings
# ---------------------------------------------------------------------
# Suppress pkg_resources deprecation warning from ultralytics
warnings.filterwarnings('ignore', category=UserWarning,
                        message='.*pkg_resources.*')

# Suppress Werkzeug development server warnings
logging.getLogger('werkzeug').setLevel(logging.ERROR)


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(BASE_DIR)

# Load environment variables from .env file in project root
env_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(env_path)

# ---------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------
STATIC_DIR = os.path.join(PROJECT_ROOT, "dynamic")
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
LABELS_FOLDER = os.path.join(STATIC_DIR, "labels")
RESULTS_FOLDER = os.path.join(STATIC_DIR, "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LABELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------
# Configure logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'app.log'))
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Flask app configuration
# ---------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))
# Validate required environment variables
if not os.getenv('GEMINI_API_KEY'):
    logger.warning("GEMINI_API_KEY not found in environment variables")
    logger.warning(f"Checked .env file at: {env_path}")
    logger.warning(
        "Please run: python3 setup_env.py to configure environment variables")

logger.info(f"Application starting from: {BASE_DIR}")
logger.info(f"Project root: {PROJECT_ROOT}")

# Bridge-related keywords for text validation
BRIDGE_KEYWORDS = [
    "bridge", "crack", "leak", "deterioration", "spalling", "joint", "deck",
    "abutment", "pier", "girder", "corrosion", "concrete", "steel", "inspection",
    "water", "damage", "repair", "maintenance", "structural", "foundation",
    "beam", "column", "slab", "pavement", "expansion", "contraction", "settlement",
    "erosion", "scaling", "efflorescence", "freeze", "thaw", "carbonation",
    "chloride", "sulfate", "alkali", "aggregate", "reinforcement", "rebar",
    "prestressed", "post-tensioned", "cable", "stay", "suspension", "arch",
    "truss", "girder", "stringer", "diaphragm", "bearing", "hinge", "expansion joint",
    "seismic", "wind", "load", "deflection", "vibration", "fatigue", "fracture",
    "delamination", "debonding", "spall", "popout", "map cracking", "shrinkage",
    "creep", "thermal", "moisture", "humidity", "precipitation", "flood", "scour",
    "undermining", "erosion", "sedimentation", "debris", "vegetation", "overgrowth"
]


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def clear_uploads():
    """
    Delete all files inside uploads, labels, and results folders.

    Utility function to clean up temporary files generated during analysis.
    Used when starting a new analysis session or resetting the application.
    Safely handles both files and subdirectories.

    Note:
        This only clears contents, not the folders themselves, as folders are created automatically if they don't exist.
    """
    for folder in [UPLOAD_FOLDER, LABELS_FOLDER, RESULTS_FOLDER]:
        if os.path.exists(folder):
            for item in os.listdir(folder):
                path = os.path.join(folder, item)
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------
# Hybrid Multi-Factor Severity Calculation
# ---------------------------------------------------------------------
def calculate_hybrid_severity(defect_name, confidence_data, detected_labels):
    """
    Calculate defect severity using hybrid multi-factor approach.

    This function combines three factors to determine severity more accurately
    than using YOLO confidence alone:

    1. YOLO Confidence Scores: Maximum confidence and detection count
       - Higher confidence indicates more certain detection
       - Multiple detections suggest widespread issue

    2. Defect Type Criticality: Engineering knowledge about defect impact
       - Bridge cracks and water leakage are inherently more critical
       - Based on safety impact from knowledge base

    3. Detection Count: Number of instances detected
       - More instances may indicate more severe problem

    The algorithm applies different thresholds based on criticality:
    - High criticality (cracks, leakage): Lower thresholds for High severity
    - Medium criticality (deterioration): Moderate thresholds
    - Low criticality (repairs): Very high thresholds required

    Args:
        defect_name: Name of the defect type (e.g., "bridge-crack")
        confidence_data: Dictionary mapping defect names to lists of confidence scores
        detected_labels: List of all detected defect names (for context)

    Returns:
        Tuple of (severity_string, css_color_class):
        - severity_string: "Low", "Medium", "High", or "Unknown"
        - css_color_class: Tailwind CSS class for UI styling (e.g., "bg-red-500")
    """
    # Normalize defect name: remove parentheses, convert to lowercase, replace spaces
    # This handles variations like "Bridge Crack", "bridge-crack", "bridge crack"
    clean_name = defect_name.split(
        '(')[0].strip().lower().replace(' ', '-').replace('*', '')

    # Special handling for "For Review" - always Unknown (grey)
    # These are low-confidence detections requiring human verification
    if clean_name == "for-review" or defect_name.lower() == "for review":
        return "Unknown", "bg-gray-500"

    # Get confidence scores for this defect type
    # confidence_data maps defect names to lists of confidence scores (one per detection)
    confidences = confidence_data.get(clean_name, [])

    # Fallback: if no confidence data, assume low severity
    if not confidences:
        return "Low", "bg-green-500"  # fallback

    # Factor 1: YOLO Confidence Metrics
    # Highest confidence among all detections
    max_confidence = max(confidences)
    # Number of times this defect was detected
    detection_count = len(confidences)

    # Factor 2: Defect Type Criticality (from knowledge base safety impact)
    # Higher criticality scores mean defects are inherently more dangerous
    # These scores are based on engineering knowledge about bridge safety
    criticality_scores = {
        "bridge-crack": 3,              # High - structural integrity risk
        "water-leakage": 3,             # High - progressive corrosion/damage
        "material-deterioration": 2,    # Medium - requires monitoring
        "repair": 1,                    # Low - informational
        "for-review": 1                 # Low - needs verification
    }

    defect_criticality = criticality_scores.get(clean_name, 1)

    # Factor 3: Apply severity thresholds based on criticality
    # Different defect types have different thresholds to account for their inherent risk
    if defect_criticality == 3:  # High criticality defects (crack, leakage)
        # Lower thresholds: even moderate confidence warrants High severity
        if max_confidence >= 0.7 or detection_count >= 3:
            return "High", "bg-red-500"
        elif max_confidence >= 0.5 or detection_count >= 2:
            return "Medium", "bg-yellow-500"
        else:
            return "Low", "bg-green-500"

    elif defect_criticality == 2:  # Medium criticality (deterioration)
        # Moderate thresholds: require higher confidence for High severity
        if max_confidence >= 0.8 and detection_count >= 3:
            return "High", "bg-red-500"
        elif max_confidence >= 0.6 or detection_count >= 2:
            return "Medium", "bg-yellow-500"
        else:
            return "Low", "bg-green-500"

    else:  # Low criticality (repair, for-review)
        # High thresholds: only very certain detections get Medium severity
        if max_confidence >= 0.9 and detection_count >= 5:
            return "Medium", "bg-yellow-500"
        else:
            return "Low", "bg-green-500"


def get_severity_from_yolo_results(detected_labels, defect_name, confidence_data):
    """
    Wrapper function to maintain compatibility with existing code.

    This function provides a backward-compatible interface to the hybrid severity calculation.
    It maintains the original function signature while using the enhanced multi-factor severity algorithm.

    Args:
        detected_labels: List of all detected defect names
        defect_name: Name of the specific defect to calculate severity for
        confidence_data: Dictionary mapping defect names to confidence scores

    Returns:
        Tuple of (severity_string, css_color_class)
    """
    return calculate_hybrid_severity(defect_name, confidence_data, detected_labels)


# ---------------------------------------------------------------------
# Parse LLM report into structured sections
# ---------------------------------------------------------------------
def parse_llm_report(llm_text: str, detected_labels: list = None, confidence_data=None):
    """
    Parse the LLM JSON response into structured sections for UI display.

    Processes the raw text response from Gemini LLM and:
    - Handles error messages gracefully
    - Extracts JSON (with fallback regex extraction if needed)
    - Validates bridge-related content
    - Adds severity scores to detected defects
    - Structures data for template rendering

    Args:
        llm_text: Raw text response from Gemini LLM (may contain JSON or error)
        detected_labels: List of defect names detected by YOLO (optional)
        confidence_data: Dictionary mapping defect names to confidence scores (optional)

    Returns:
        Dictionary with structured report sections and defects, or error message
        if parsing fails or content is invalid.
    """
    if not detected_labels:
        detected_labels = []

    print("DEBUG: LLM Response:", repr(llm_text))  # Debug print

    # Check if the response is an error message from generate_bridge_report
    # Handle API errors (quota exceeded, invalid key, etc.) with user-friendly messages
    if llm_text and llm_text.strip().startswith("(Gemini error:"):
        error_msg = llm_text.strip()
        # Extract a more user-friendly message
        if "quota" in error_msg.lower() or "429" in error_msg:
            return {"valid": False, "message": "API quota exceeded. Please check your Gemini API key or try again later."}
        elif "api key" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            return {"valid": False, "message": "Invalid or missing API key. Please check your Gemini API key configuration."}
        else:
            return {"valid": False, "message": f"Error generating report: {error_msg[:100]}..."}

    # Attempt to parse JSON response
    try:
        parsed = json.loads(llm_text)
    except json.JSONDecodeError:
        # Fallback: Try to extract JSON from text if LLM wrapped it in markdown or other text
        # This handles cases where LLM might add explanatory text around the JSON
        json_match = re.search(r'\{.*\}', llm_text, re.DOTALL)
        if json_match:
            print("DEBUG: Extracted JSON:", repr(
                json_match.group(0)))  # Debug print
            try:
                parsed = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return {"valid": False, "message": "Invalid response format. Please try again."}
        else:
            return {"valid": False, "message": "Invalid response format. Please try again."}

    # Validate that content is bridge-related
    # LLM may reject non-bridge images/text and return is_bridge_related: false
    if not parsed.get('is_bridge_related', True):
        return {"valid": False, "message": "Please provide input related to bridge defects or inspection."}

    # Process defects: add severity scores and color coding
    defects = parsed.get('detailed_defects', [])
    # For text-only inputs or images with no YOLO detections, don't show defect list
    if confidence_data is None or len(confidence_data) == 0:
        defects = []
    else:
        # Add severity rating and UI color for each detected defect
        # Severity is calculated using hybrid approach (confidence + criticality + count)
        for i, defect in enumerate(defects):
            name = defect.get('name', '')
            # Calculate severity using multi-factor approach
            severity, severity_color = get_severity_from_yolo_results(
                detected_labels, name, confidence_data)
            defect['severity'] = severity  # Low, Medium, High, or Unknown
            # CSS class for UI styling
            defect['severity_color'] = severity_color
            defect['number'] = str(i + 1)  # Sequential numbering for display

    # Remove detailed_defects from sections to avoid duplication
    # We extract it to the defects array above with severity added
    sections_clean = {k: v for k, v in parsed.items() if k !=
                      'detailed_defects'}

    # Return structured results
    return {
        "sections": sections_clean,
        "defects": defects
    }


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
# Flask route decorator and session handling patterns
# References:
#  https://flask.palletsprojects.com/en/stable/api/#sessions
#  https://flask.palletsprojects.com/en/stable/api/
@app.route("/", methods=["GET"])
def index():
    """
    Render the main landing page of the web application.

    This route displays the initial form where users can upload an image or enter a text description. 
    It preserves any text input from previous attempts to allow users to retry with the same input if there was an error.

    Returns:
        Rendered HTML template (index.html) with empty results and any preserved text input from previous attempts.
    """
    # Preserve text input for retry in case of previous errors
    text_input = session.pop('text_input', None)
    return render_template("index.html", results=None, image_path=None, labelled_image=None, text_input=text_input)

# Flask file upload and request handling patterns
# References:
#  https://flask.palletsprojects.com/en/3.0.x/patterns/fileuploads/


@app.route("/", methods=["POST"])
def analyze():
    """
    Main analysis endpoint that processes user input (image or text).

    This is the core function that handles two types of input:
    1. Image upload: Runs YOLO detection, then LLM analysis
    2. Text description: Validates bridge-related keywords, then LLM analysis

    Workflow:
    - Validates input type (image file or text)
    - For images: Validates file type, runs YOLO detection, extracts defects
    - Processes through appropriate pipeline (YOLO+LLM or LLM-only)
    - Generates structured JSON report using Gemini LLM
    - Calculates hybrid severity scores for detected defects
    - Saves results and redirects to results page

    Returns:
        Redirect to results page on success, or back to index on error.
        Errors are logged and user-friendly messages are displayed.

    Raises:
        Various exceptions during YOLO/LLM processing are caught and handled
        gracefully with user-friendly error messages flashed to the UI.
    """
    try:
        # Extract input from form (either image file or text description)
        file = request.files.get("image")
        text = request.form.get("defect_description")

        # Store inputs for retry
        session['text_input'] = text.strip() if text else None

        image_filename = None
        results = None
        labelled_image_path = None

        # --------------------------
        # Image upload + YOLO run
        # --------------------------
        if file and file.filename != "":
            logger.info(f"Processing image upload: {file.filename}")

            if not file.mimetype.startswith("image/"):
                logger.warning(f"Invalid file type: {file.mimetype}")
                flash("Only image files are allowed.", "error")
                return redirect(url_for("index"))

            filename = f"{secrets.token_hex(8)}_{file.filename}"
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)
            logger.info(f"Image saved to: {image_path}")

            # Run YOLO and store labeled image
            try:
                labelled_filename, yolo_results = run_yolo(
                    image_path, LABELS_FOLDER)
                detected_labels = []
                confidence_data = {}  # JSON serializable confidence data
                logger.info(f"YOLO detection completed: {labelled_filename}")
            except Exception as e:
                logger.error(f"YOLO detection failed: {str(e)}")
                flash(
                    "Error processing image with YOLO model. Please try again.", "error")
                return redirect(url_for("index"))

            if labelled_filename:
                labelled_image_path = f"labels/{labelled_filename}"
                label_summary = os.path.splitext(labelled_filename)[0]

                # Extract detected labels and confidence data from YOLO results object
                if yolo_results and len(yolo_results.boxes) > 0:
                    # YOLO class ID to defect name mapping
                    # Maps numeric class IDs (0-4) to human-readable defect names
                    yolo_class_map = {
                        0: "bridge-crack",              # Class 0: Bridge cracks
                        1: "for-review",                # Class 1: Areas needing manual review
                        2: "material-deterioration",    # Class 2: Material degradation
                        3: "repair",                    # Class 3: Repair areas
                        4: "water-leakage"              # Class 4: Water leakage
                    }

                    # Extract unique defect types detected (remove duplicate classes)
                    # Convert class IDs to integers, get unique set
                    unique_classes = set(int(cls)
                                         for cls in yolo_results.boxes.cls)
                    # Map class IDs to defect names, filter valid mappings
                    detected_labels = [yolo_class_map[cls_id]
                                       for cls_id in unique_classes if cls_id in yolo_class_map]
                    detected_labels = list(
                        set(detected_labels))  # Remove duplicates

                    # Extract confidence scores for each detection instance
                    # Store as list per defect type for severity calculation
                    for i, cls in enumerate(yolo_results.boxes.cls):
                        class_id = int(cls)
                        # Confidence score (0.0-1.0)
                        conf = float(yolo_results.boxes.conf[i])
                        class_name = yolo_class_map[class_id]

                        # Group confidence scores by defect type
                        if class_name not in confidence_data:
                            confidence_data[class_name] = []
                        confidence_data[class_name].append(conf)
            else:
                label_summary = "No defects detected"

            # Generate LLM report using Gemini API
            try:
                # Determine which image to send to LLM based on detection results
                if detected_labels:
                    # Case 1: YOLO detected defects - send labeled image with bounding boxes
                    # This helps LLM focus on detected areas and provides visual context
                    image_for_llm = os.path.join(
                        LABELS_FOLDER, labelled_filename) if labelled_filename else None
                else:
                    # Case 2: No YOLO detections - send original image for LLM validation
                    # LLM can still analyze the image to confirm it's bridge-related
                    image_for_llm = image_path

                gemini_text = generate_bridge_report(
                    label_summary,
                    image_for_llm,
                    detected_labels  # Pass detected labels directly
                )
                logger.info("LLM report generated successfully")
            except Exception as e:
                logger.error(f"LLM report generation failed: {str(e)}")
                flash(
                    "Error generating analysis report. Please check your API key and try again.", "error")
                return redirect(url_for("index"))

            # Parse structured report into sections
            parsed_report = parse_llm_report(
                gemini_text, detected_labels, confidence_data)

            results = parsed_report

            # Save results to a JSON file instead of session cookie
            results_path = os.path.join(RESULTS_FOLDER, "results.json")
            with open(results_path, "w") as f:
                json.dump(results, f)

            # Keep only small references in session
            session["image_filename"] = filename
            session["labelled_image"] = labelled_image_path
            session["results_path"] = results_path
            # Store JSON serializable confidence data
            session["confidence_data"] = confidence_data

            return redirect(url_for("results"))

        # --------------------------
        # Text-only input
        # --------------------------
        elif text and text.strip() != "":
            text_clean = text.strip()
            print(f"SERVER LOG: Processing text input: '{text_clean}'")

            # Validate text contains bridge-related keywords
            # Check if input contains any engineering/bridge terminology to filter
            # non-relevant inputs before expensive LLM API calls
            matching_keywords = [
                kw for kw in BRIDGE_KEYWORDS if kw in text_clean.lower()]
            print(f"SERVER LOG: Matching keywords: {matching_keywords}")

            if not matching_keywords:
                # No bridge-related keywords found - reject without LLM call
                print("SERVER LOG: Text validation failed - no bridge keywords found")
                results = {
                    "valid": False, "message": "Please provide input related to bridge defects or inspection.", "original_input": text_clean}
            else:
                # Validation passed - proceed with LLM analysis
                print("SERVER LOG: Text validation passed - calling LLM")
                # Gemini report for text
                gemini_text = generate_bridge_report(
                    text_clean, image_path=None)
                print(f"SERVER LOG: LLM raw response: {repr(gemini_text)}")

                # Parse structured report
                parsed_report = parse_llm_report(
                    gemini_text, detected_labels=[], confidence_data=None)
                print(f"SERVER LOG: Parsed report: {parsed_report}")

                results = parsed_report
                # Add original input for display
                results['original_input'] = text_clean

                # Check if overall_severity is present
                if 'sections' in results and 'overall_severity' not in results['sections']:
                    print(
                        "SERVER LOG: WARNING - overall_severity missing from LLM response")
                else:
                    print(
                        f"SERVER LOG: overall_severity found: {results.get('sections', {}).get('overall_severity', 'N/A')}")

            # Save results
            results_path = os.path.join(RESULTS_FOLDER, "results.json")
            with open(results_path, "w") as f:
                json.dump(results, f)

            session["results_path"] = results_path
            return redirect(url_for("results"))

        else:
            logger.warning("No input provided")
            flash("Please upload an image or enter a text description.", "error")
            return redirect(url_for("index"))

    except Exception as e:
        logger.error(
            f"Unexpected error in analyze route: {str(e)}", exc_info=True)
        flash("An unexpected error occurred. Please try again.", "error")
        return redirect(url_for("index"))


@app.route("/results")
def results():
    """
    Display the analysis results page.

    Retrieves previously saved analysis results from the session and displays them to the user. 
    Results include detected defects, LLM-generated report, and visual annotations (if image was uploaded).

    Returns:
        Rendered HTML template with analysis results, or redirects to index if no results are found in session.

    Note:
        Results are stored as JSON files and referenced via session to avoid storing large data in session cookies.
    """
    # Retrieve result references from session
    results_path = session.get("results_path")
    image_filename = session.get("image_filename")
    labelled_image = session.get("labelled_image")

    # Validate that results exist
    if not results_path or not os.path.exists(results_path):
        flash("No analysis found. Please upload or describe a defect first.", "error")
        return redirect(url_for("index"))

    # Load results from JSON file
    with open(results_path) as f:
        results = json.load(f)

    # Construct image path for display (if image was uploaded)
    image_path = f"uploads/{image_filename}" if image_filename else None

    return render_template("index.html", results=results, image_path=image_path, labelled_image=labelled_image)


@app.route("/startover")
def start_over():
    """
    Reset the application state and clear all uploaded files.

    This route clears:
    - All uploaded images and generated labels from dynamic folders
    - Session data (results, image references, etc.)

    Useful for starting a fresh analysis session.

    Returns:
        Redirect to the main index page.
    """
    clear_uploads()  # Delete all files in uploads, labels, and results folders
    session.clear()  # Clear all session data
    return redirect(url_for("index"))


# ---------------------------------------------------------------------
# Static routes for serving files
# ---------------------------------------------------------------------
# Flask send_from_directory for static file serving
# References:
#  https://flask.palletsprojects.com/en/3.0.x/api/#flask.send_from_directory

@app.route("/uploads/<path:filename>")
def serve_uploaded_file(filename):
    """
    Serve uploaded image files to the web browser.

    Flask route that allows the web interface to display original uploaded images. Files are served from the UPLOAD_FOLDER directory.

    Args:
        filename: Name of the file to serve (from URL path)

    Returns:
        File response with the requested image file.
    """
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/labels/<path:filename>")
def serve_labelled_file(filename):
    """
    Serve YOLO-labeled images with bounding boxes to the web browser.

    Flask route that serves images with defect annotations (bounding boxes and labels) generated by YOLO detection. 
    These images are displayed in the results page to show detected defects visually.

    Args:
        filename: Name of the labeled image file to serve (from URL path)

    Returns:
        File response with the requested labeled image file.
    """
    return send_from_directory(LABELS_FOLDER, filename)


# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Get configuration from environment variables with sensible defaults

    # Host Configuration: '0.0.0.0' allows Flask to accept connections from all network interfaces
    #   This is required for Docker (to accept connections from outside container)
    #   This also works for local development (accepts localhost connections)
    # - 'localhost' or '127.0.0.1' would only work locally, NOT in Docker
    host = os.getenv('FLASK_HOST', '0.0.0.0')

    # Port Configuration: Default: 5000 (works for Windows and most systems)
    port = int(os.getenv('FLASK_PORT', '5001'))

    # Suppress Werkzeug's verbose startup messages for cleaner console output
    # References: https://flask.palletsprojects.com/en/3.0.x/logging/#werkzeug-logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    print(f"Access the application at: http://localhost:{port}")

    app.run(host=host, port=port, debug=True, use_reloader=False)
