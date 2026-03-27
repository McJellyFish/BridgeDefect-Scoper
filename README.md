================================================================================
MULTIMODAL BRIDGE DEFECT SCOPER
================================================================================

Author(s): Junjie Ma, Tanmay Sahasrabudhe, Sanchana Mohnraj
Date: 03/26/2026
Course: AIT 626 NLP - Natural Language Processing
Project: Multimodal Bridge Defect Scoper
Institution: George Mason University

================================================================================
IMPORTANT: FOLDER STRUCTURE UPDATE
================================================================================

The evaluation dataset folder structure has been reorganized for clarity:

OLD STRUCTURE (Previous Versions):

- evaluation/ (ground truth data)
- evaluation_new/ (generated predictions)

NEW STRUCTURE (Current Version):

- evaluation_data/
  ├── ground_truth/ (reference data for evaluation)
  └── predictions/ (generated reports for evaluation)

All evaluation scripts have been updated to use the new structure.
The new structure is more intuitive:

- "ground_truth" clearly indicates reference data
- "predictions" clearly indicates generated outputs
- Both are organized under "evaluation_data" parent folder

If you have an older version with "evaluation/" and "evaluation_new/" folders,
please rename them to match the new structure, or set environment variables:

export EVAL_GROUND_TRUTH_DIR=/path/to/evaluation
export EVAL_PREDICTIONS_DIR=/path/to/evaluation_new

================================================================================

1. # PROBLEM STATEMENT

Bridge inspection is a critical infrastructure maintenance task that requires expert analysis to identify structural defects.
Manual inspection is:

- Time-consuming and labor-intensive
- Subject to human error and inconsistency
- Requires domain expertise not always available
- Limited in scalability for large infrastructure networks

This project addresses these challenges by combining computer vision and natural language processing to automate bridge defect detection and assessment.
The system can process both visual (images) and textual (descriptions) inputs, making it accessible to field inspectors with varying technical capabilities.

Traditional inspection methods rely heavily on manual visual assessment, which can miss subtle defects, vary in quality between inspectors, and create extensive documentation overhead.
The integration of AI-powered detection with natural language generation enables consistent, comprehensive defect analysis and reporting.

# ================================================================================ 2. SOLUTION APPROACH

Our solution combines YOLO object detection with Google's Gemini LLM to create a multimodal bridge defect analysis system.

Point-by-Point Solution Outline:

a) Image Input Processing

- Users upload bridge inspection images
- YOLO model (custom-trained) detects defects with bounding boxes
- Five defect classes: bridge-crack, water-leakage, material-deterioration, repair, for-review
- Confidence scores extracted for each detection

b) Text Input Processing

- Users can provide text descriptions of bridge defects
- Keyword-based validation ensures bridge-related content
- 60+ bridge-related keywords filter non-relevant inputs

c) LLM Analysis and Report Generation

- Google Gemini 2.5 Flash processes detected defects or text descriptions
- Knowledge base provides engineering context for each defect type
- Structured JSON report generated with:
  - Defect summaries
  - Engineering assessments
  - Safety risks
  - Recommended actions
  - Overall severity rating

d) Hybrid Severity Calculation

- Multi-factor approach combining:
  - YOLO confidence scores (max and average)
  - Defect type criticality (from knowledge base)
  - Detection count (number of instances)
- Maps to Low/Medium/High severity levels

e) Web Interface

- Flask-based web application
- Visual display of detected defects with bounding boxes
- Formatted engineering reports
- User-friendly interface for non-technical users

# ================================================================================ 3. SYSTEM ARCHITECTURE

## Workflow:

User Input (Image or Text)
↓
[If Image] YOLO Object Detection → Detected Defects + Confidence Scores
↓
Gemini LLM Analysis (with Knowledge Base Context)
↓
Structured Report Generation (JSON)
↓
Hybrid Severity Calculation
↓
Web UI Display (Formatted Report + Visual Annotations)

## Components:

1. YOLO Model (scripts/best.pt)
   - Custom-trained on bridge defect dataset from Roboflow
   - 5 classes, confidence threshold: 0.25, IOU: 0.7

2. LLM Module (llmreport_demo.py)
   - Google Gemini 2.5 Flash API
   - Knowledge base with defect-specific engineering information
   - Structured prompt engineering for consistent JSON output

3. Web Application (app_demo.py)
   - Flask framework
   - File upload handling
   - Session management
   - Result visualization

4. Evaluation Framework
   - Ground truth dataset (143 images + 146 text cases)
   - Objective metrics (classification accuracy, completeness)
   - LLM-as-Judge metrics (DeepEval: faithfulness, relevancy, hallucination)

# ================================================================================ 4. EXAMPLE INPUT AND OUTPUT

## Example 1: Image Input

Input:

- Image file: bridge_inspection_001.jpg (uploaded via web interface)

YOLO Detection Output:

- bridge-crack: 2 detections (conf: 0.85, 0.72)
- material-deterioration: 1 detection (conf: 0.68)
- water-leakage: 1 detection (conf: 0.91)

LLM Report Output:
{
"is_bridge_related": true,
"defect_summary": "The bridge structure exhibits widespread material deterioration,
indicating long-term exposure to environmental factors. Multiple
instances of water leakage are present, which are a primary
accelerator of further degradation. Several bridge cracks have been
identified, some requiring immediate attention...",
"resolution_summary": "Immediate actions should focus on identifying and sealing water
leakage sources to prevent further moisture ingress and corrosion.
All identified cracks need to be mapped, measured, and sealed...",
"detailed_defects": [
{
"name": "bridge-crack",
"engineering_assessment": "Visible fractures or splits in bridge components are
critical indicators of tension or structural stress...",
"safety_risks": "Cracks may lead to spalling, delamination, or even localized
collapse if left untreated...",
"recommended_actions": "Precisely map and measure crack widths and depths.
Conduct structural integrity testing as needed...",
"severity": "High"
}
],
"further_recommendations": "Given the prevalence of material deterioration and water
leakage, a comprehensive assessment of the bridge's
waterproofing system is strongly recommended..."
}

## Example 2: Text Input

Input Text:
"Large crack observed on bridge deck near pier 3. Significant water seepage visible.
Concrete spalling noted in multiple locations."

LLM Report Output:
{
"is_bridge_related": true,
"defect_summary": "Multiple defects observed including cracking, water leakage,
and spalling. These conditions indicate significant structural
concerns requiring immediate assessment...",
"resolution_summary": "Comprehensive inspection and immediate repair action required.
Water leakage must be addressed first to prevent further
deterioration...",
"overall_severity": "High",
"further_recommendations": "Engage structural engineer for detailed assessment.
Consider temporary traffic restrictions if warranted..."
}

5. # INSTALLATION AND SETUP
   Prerequisites:

---

- Python 3.8 or higher
  - Verify version: python3 --version
- pip package manager
- Internet connection (for API calls)

## Step 1: Extract and Navigate to Project Directory

1. Extract project files (if downloaded as archive)
2. Navigate to project directory:
   cd Project

## Step 2: Create Virtual Environment (Recommended)

Creating a virtual environment isolates project dependencies:

python3 -m venv venv

Activate the virtual environment:

On Mac/Linux:
source venv/bin/activate

On Windows:
venv\Scripts\activate

After activation, you should see (venv) in your terminal prompt.

Verify Python version:
python --version # Should show Python 3.8 or higher

## Step 3: Install Dependencies

Install all required Python packages:

pip install -r requirements.txt

This will install all necessary packages including:

- Flask (web framework)
- google-generativeai (Gemini API)
- ultralytics (YOLO object detection)
- torch, torchvision (PyTorch)
- opencv-python (image processing)
- numpy, pandas, matplotlib, seaborn (data processing)
- scikit-learn (evaluation metrics)
- deepeval, openai (LLM evaluation)
- And other dependencies (see requirements.txt for complete list)

Note: Installation may take 5-10 minutes depending on internet speed.

## Step 4: Configure Environment Variables (If Needed)

The .env file is included in the submission with API keys pre-configured for testing.

If you need to update API keys (e.g., if provided keys have expired):

1. Open .env file in project root directory
2. Update the following values if needed:
   GEMINI_API_KEY=your-gemini-api-key-here
   OPEN_AI_API_KEY=your-openai-api-key-here # Optional, for DeepEval
   FLASK_SECRET_KEY=your-secret-key-here

Note: The included .env file should work for testing. To obtain new API keys:

- Gemini: https://makersuite.google.com/app/apikey
- OpenAI: https://platform.openai.com/api-keys (optional, for DeepEval only)

## Step 5: Verify Installation

Test that all dependencies are correctly installed:

python3 -c "from dotenv import load_dotenv; from flask import Flask; from ultralytics import YOLO; import google.generativeai as genai; print('All imports successful!')"

If you see "All imports successful!", installation is complete.

## Step 6: Verify YOLO Model File

Ensure the trained YOLO model exists:

ls scripts/best.pt # Should show the model file

The file scripts/best.pt should be present in the scripts directory.

## Step 7: Download Evaluation Dataset (Optional)

If you plan to run evaluation scripts, download the evaluation dataset.
See Section 6.2 for detailed download instructions.

# ================================================================================ 6. DATASET INFORMATION

## 6.1 Training Dataset

The YOLO model was trained on a bridge defect detection dataset from Roboflow.

Dataset Details:

- Source: Roboflow Universe
- Download Link: https://universe.roboflow.com/xltv/bridge-defects-detection-mtlxt-vkuos/dataset/1
- Classes: 5 defect types
  - bridge-crack
  - for-review
  - material-deterioration
  - repair
  - water-leakage

Download Instructions:

1. Visit: https://universe.roboflow.com/xltv/bridge-defects-detection-mtlxt-vkuos/dataset/1
2. Sign up or log in to Roboflow
3. Click "Download" and select "YOLO v11" format
4. Extract dataset following YOLO training structure

Note: Pre-trained model weights (scripts/best.pt) are included.

## 6.2 Evaluation Dataset (Download Required)

Unfortunately, evaluation cannot be done directly from docker, below are the steps provided for manual evaluation.

IMPORTANT: The dataset uses the NEW folder structure:

- evaluation_data/ground_truth/ (not "evaluation/")
- evaluation_data/predictions/ (not "evaluation_new/")

Download Instructions:

1. Download from: https://drive.google.com/drive/folders/1tiQHzSpUkXYCh0cOhp4qu7V0Cv_wwsN7?usp=drive_link
2. Extract to project root:
   Project/
   └── evaluation_data/ # Extract here
   ├── ground_truth/
   └── predictions/

3. Verify structure:
   Project/
   ├── scripts/
   ├── evaluation_data/ # Downloaded dataset
   │ ├── ground_truth/
   │ │ ├── case1/
   │ │ ├── case2/
   │ │ ├── ground_truth.csv
   │ │ └── results/
   │ └── predictions/
   │ ├── case1/
   │ └── case2/
   └── ...

Alternative Location:
If extracting to different location, set environment variables:
export EVAL_GROUND_TRUTH_DIR=/path/to/ground_truth
export EVAL_PREDICTIONS_DIR=/path/to/predictions

Dataset Contents:

- Ground Truth: 143 image cases + 146 text cases with reference reports
- Format: JSON reports, YOLO labels (.txt), original images

  6.3 Model Weights

---

Pre-trained YOLO Model:

- File: scripts/best.pt
- Trained on: ~13k images from Roboflow dataset
- Format: PyTorch YOLO v11 weights

# ================================================================================ 7. RUNNING THE SYSTEM

## 7.1 Running with Docker (Easiest - No Installation Required)

Docker provides the easiest way to run the system without installing dependencies.

Prerequisites:

- Docker Desktop installed on your system
- Download Docker Desktop from: https://www.docker.com/products/docker-desktop/

Steps:

1. Install Docker Desktop (if not already installed):
   - Download from: https://www.docker.com/products/docker-desktop/
   - Install and launch Docker Desktop
   - Ensure Docker Desktop is running

2. Pull the Docker image:
   - Open Docker Desktop
   - Go to Images tab
   - Search for: xltv/bridge_scoper:v1.0
   - Click "Pull" button

   OR use command line:
   docker pull xltv/bridge_scoper:v1.0

3. Run the container:

   For Windows users:
   docker run -p 5000:5000 xltv/bridge_scoper:v1.0

   For Mac users:
   On macOS, port 5000 is reserved by a system process called ControlCenter, which is used for system features such as AirPlay, Screen Mirroring, and Handoff.
   Because this system service always occupies port 5000, it cannot be released or reassigned to Docker containers.
   Therefore, when running our Flask application in Docker, we cannot map the container’s internal port 5000 directly to port 5000 on macOS.
   Instead, we map it to another available port such as 8000.

   docker run -p 8000:5000 xltv/bridge_scoper:v1.0

4. Access the application:
   - Windows: Open browser to http://127.0.0.1:5000
   - Mac: Open browser to http://127.0.0.1:8000

5. Use the interface:
   - Upload image OR enter text description
   - Click "Analyze Defect"
   - View results with detected defects and report

6. Stop the container:
   - Press Ctrl+C in the terminal where container is running
   - OR find container in Docker Desktop and click "Stop"

Note: The Docker image includes all dependencies and the application pre-configured. No Python installation or dependency management required.

NO evaluation can be done using docker, dataset has been shared for manual evaluation. Also, user cannot notice the dynamic images added to the folder while analysis Because they are in docker container. To test these features need to run manually.

## 7.2 Running from Source Code

If you prefer to run from source code or need to modify the code:

1. Navigate to scripts directory:
   cd scripts

2. Run Flask application:
   python app_demo.py

3. Open web browser using the link given.

4. Use the interface:
   - Upload image OR enter text description
   - Click "Analyze Defect"
   - View results with detected defects and report

Note: Ensure you have completed Section 5 (Installation and Setup) before running.

Please kill the current session with the ports in order to re run or if u face any issues with port.

## 7.3 Running Evaluation (Optional)

If you have downloaded the evaluation dataset:

Step 1: Generate Prediction Reports
cd scripts
python generate_eval_reports.py
(This processes all images/text and generates reports)

Step 2: Run Objective Evaluation
python evaluate_llm_reports.py
(Generates evaluation_results.csv and plots)

Step 3: Run DeepEval (Optional, requires OpenAI API key)
python evaluate_with_deepeval.py
(Generates LLM-as-Judge metrics)

Results will be saved to:
evaluation_data/ground_truth/results/
├── plots/ # Objective evaluation plots
├── deepeval_plots/ # DeepEval visualization plots
└── \*.csv # Evaluation results in CSV format

# ================================================================================ 8. PROJECT STRUCTURE

Project/
├── scripts/
│ ├── app_demo.py # Main Flask web application
│ ├── llmreport_demo.py # LLM report generation module
│ ├── yolo_model.py # YOLO model wrapper
│ ├── config.py # Path configuration
│ ├── generate_eval_reports.py # Batch report generator
│ ├── evaluate_llm_reports.py # Objective evaluation
│ ├── evaluate_with_deepeval.py # LLM-as-Judge evaluation
│ ├── best.pt # Trained YOLO model weights
│ └── templates/
│ └── index.html # Web UI template
│
├── evaluation_data/ # Evaluation dataset (download separately)
│ ├── ground_truth/ # Reference data
│ │ ├── case1/ # Image test cases (143 images)
│ │ │ ├── original_images/ # Original test images
│ │ │ ├── labels/ # YOLO detection labels
│ │ │ ├── labelled_images/ # Visualized labels
│ │ │ └── reports/ # Ground truth JSON reports
│ │ ├── case2/ # Text test cases (146 inputs)
│ │ │ ├── inputs/ # Text input files
│ │ │ └── reports/ # Ground truth JSON reports
│ │ ├── ground_truth.csv # Metadata CSV file
│ │ └── results/ # Evaluation outputs
│ │ ├── plots/ # Visualization plots
│ │ └── deepeval_plots/ # DeepEval visualization plots
│ └── predictions/ # Generated reports
│ ├── case1/
│ │ ├── labeled_images/ # YOLO-labeled images
│ │ └── reports/ # LLM-generated reports
│ └── case2/
│ └── reports/ # LLM-generated reports
│
├── dynamic/ # Runtime folders (auto-created)
│ ├── uploads/ # User-uploaded images
│ ├── labels/ # YOLO-labeled images
│ └── results/ # Analysis results
│
├── requirements.txt # Python dependencies
├── README.txt  
<<<<<<< HEAD
└── .env # Environment variables found in hidden files
=======
└── .env # Environment variables found in hidden files

> > > > > > > 835f1921b44712b24a058a68da9cb846e1f9c8e4
