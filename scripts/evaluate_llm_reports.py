#!/usr/bin/env python3
"""
Comprehensive LLM Report Evaluation

Author(s): Sai Praneetha, Junjie Ma, I Chen Sung
Date: 12/03/2025
Course: AIT 526 NLP - Natural Language Processing
Project: Multimodal Bridge Defect Scoper
Institution: George Mason University

Evaluates LLM-generated reports against automated ground truth using objective metrics.
"""

import os
import json
import csv
import sys
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Paths - using centralized config
from config import (
    GROUND_TRUTH_CSV,
    CASE1_PREDICTION_REPORTS,
    CASE2_PREDICTION_REPORTS,
    EVALUATION_RESULTS_CSV,
    EVALUATION_SUMMARY_TXT,
    PLOTS_DIR as PLOTS_DIR_PATH
)

# Convert Path objects to strings for compatibility
GROUND_TRUTH_PATH = str(GROUND_TRUTH_CSV)
CASE1_REPORTS = str(CASE1_PREDICTION_REPORTS)
CASE2_REPORTS = str(CASE2_PREDICTION_REPORTS)
RESULTS_CSV = str(EVALUATION_RESULTS_CSV)
SUMMARY_TXT = str(EVALUATION_SUMMARY_TXT)
PLOTS_DIR = str(PLOTS_DIR_PATH)


def load_ground_truth():
    """
    Load ground truth CSV file into a dictionary for evaluation.
    
    Reads the ground_truth.csv file which contains reference data for each test case including:
    - Whether the input is bridge-related
    - YOLO-detected defects (for image cases)
    - Test case category (image or text)
    
    The function processes CSV data by:
    - Converting string boolean values to Python booleans
    - Parsing pipe-delimited YOLO detections into lists
    
    Returns:
        Dictionary mapping test_id (e.g., "case1_1", "case2_01") to ground truth data dictionary with keys:
        - test_id: Unique identifier
        - category: 'image' or 'text'
        - is_bridge_related: Boolean or None
        - yolo_detections: Pipe-delimited string of defect names
        - yolo_detections_list: List of defect names
    """
    ground_truth = {}
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_id = row['test_id']
            # Convert string booleans ("True"/"False") to Python booleans
            # CSV stores booleans as strings, need to convert for comparison
            if row['is_bridge_related'] == 'True':
                row['is_bridge_related'] = True
            elif row['is_bridge_related'] == 'False':
                row['is_bridge_related'] = False
            else:
                row['is_bridge_related'] = None  # Unknown/missing value
            
            # Convert YOLO detections from pipe-delimited string to list
            # Format in CSV: "bridge-crack|water-leakage|material-deterioration"
            if row['yolo_detections'] and row['yolo_detections'] != 'N/A':
                row['yolo_detections_list'] = row['yolo_detections'].split('|')
            else:
                row['yolo_detections_list'] = []  # No detections or N/A
            
            ground_truth[test_id] = row
    
    return ground_truth


def load_report(test_id):
    """
    Load a JSON report file for a given test case ID.
    
    Determines the correct report file path based on test_id format:
    - case1_<num>: Image test cases (reports in case1/reports/)
    - case2_<num>: Text test cases (reports in case2/reports/)
    
    Args:
        test_id: Test case identifier (e.g., "case1_1", "case2_01")
    
    Returns:
        Dictionary containing parsed JSON report data, or error dictionary if the file doesn't exist or JSON parsing fails.
        Error format: {'error': '<error_message>'}
    """
    # Determine report file path based on test_id prefix
    if test_id.startswith('case1_'):
        # Case 1: Image test cases
        img_num = test_id.split('_')[1]  # Extract image number
        report_path = os.path.join(CASE1_REPORTS, f"{img_num}.json")
    elif test_id.startswith('case2_'):
        # Case 2: Text test cases
        text_num = test_id.split('_')[1]  # Extract text case number
        report_path = os.path.join(CASE2_REPORTS, f"{text_num}.json")
    else:
        return None  # Invalid test_id format
    
    # Check if report file exists
    if not os.path.exists(report_path):
        return None  # Report file not found
    
    # Load and parse JSON report
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        # Return error dictionary if JSON parsing fails
        return {'error': str(e)}


def evaluate_classification(report, gt_is_bridge_related):
    """
    Evaluate binary classification accuracy: is the input bridge-related?
    
    This metric measures whether the LLM correctly identifies if the input (image or text) is related to bridge inspection.
    This is a fundamental validation metric to ensure the system doesn't process irrelevant inputs.
    
    Args:
        report: Dictionary containing LLM-generated report (may have different structures)
        gt_is_bridge_related: Ground truth boolean value (True/False/None)
    
    Returns:
        Tuple of (predicted_value, is_correct, error_message):
        - predicted_value: Boolean prediction from LLM, or None if missing
        - is_correct: Boolean indicating if prediction matches ground truth
        - error_message: String describing any errors, or None if successful
    
    Note:
        The function handles different report structures because reports may be stored directly or nested under 'sections' key depending on format.
    """
    if report is None:
        return None, False, "Report missing"
    
    if 'error' in report:
        return None, False, f"JSON error: {report['error']}"
    
    # Extract is_bridge_related field from different possible report structures
    # Reports may have different formats depending on parsing/processing
    predicted = None
    if 'is_bridge_related' in report:
        # Direct field in report
        predicted = report['is_bridge_related']
    elif 'sections' in report and 'is_bridge_related' in report['sections']:
        # Nested under 'sections' key
        predicted = report['sections']['is_bridge_related']
    elif 'valid' in report:
        # For Case 2 text rejections, valid=False implies not bridge-related
        predicted = False if report['valid'] == False else None
    
    # Validate that prediction was extracted
    if predicted is None:
        return None, False, "is_bridge_related field missing"
    
    # Compare prediction with ground truth
    is_correct = (predicted == gt_is_bridge_related)
    return predicted, is_correct, None


def evaluate_defect_completeness(report, gt_defects):
    """
    Evaluate if LLM report includes all YOLO-detected defects.
    
    This metric measures how completely the LLM report covers the defects detected by YOLO.
    A completeness score of 1.0 means all YOLO-detected defects are mentioned in the LLM report. This is important because:
    - Missing defects could lead to incomplete inspections
    - Extra defects might indicate hallucinations
    
    Completeness Formula: (matched_defects / total_yolo_defects) * 100
    
    Args:
        report: Dictionary containing LLM-generated report
        gt_defects: List of defect names detected by YOLO (ground truth)
    
    Returns:
        Tuple of (completeness_score, llm_defects, missing_defects, extra_defects):
        - completeness_score: Float (0.0-1.0), or None if no ground truth defects
        - llm_defects: List of defect names mentioned in LLM report
        - missing_defects: List of YOLO-detected defects NOT in LLM report
        - extra_defects: List of defects in LLM report NOT detected by YOLO
    
    Note:
        Defect names are normalized (lowercase, stripped) for comparison to handle variations like "Bridge Crack" vs "bridge-crack" vs "bridge crack".
    """
    # No ground truth defects - completeness not applicable
    if not gt_defects:
        return None, [], [], []
    
    # Report missing or has errors - completeness is 0
    if report is None or 'error' in report:
        return 0.0, [], gt_defects, []
    
    # Extract defect names from LLM report
    # Handle different report structures (direct vs nested under 'sections')
    llm_defects = []
    if 'detailed_defects' in report:
        for defect in report['detailed_defects']:
            name = defect.get('name', '').lower().strip()
            llm_defects.append(name)
    elif 'sections' in report and 'detailed_defects' in report['sections']:
        for defect in report['sections']['detailed_defects']:
            name = defect.get('name', '').lower().strip()
            llm_defects.append(name)
    
    # Normalize defect names for case-insensitive comparison
    # Handles variations: "Bridge-Crack" vs "bridge-crack" vs "bridge crack"
    gt_defects_normalized = [d.lower().strip() for d in gt_defects]
    llm_defects_normalized = [d.lower().strip() for d in llm_defects]
    
    # Calculate completeness score: percentage of YOLO defects mentioned in report
    matched = sum(1 for gt_d in gt_defects_normalized if gt_d in llm_defects_normalized)
    completeness = matched / len(gt_defects_normalized) if gt_defects_normalized else 1.0
    
    # Identify missing defects (detected by YOLO but not in LLM report)
    missing = [d for d in gt_defects_normalized if d not in llm_defects_normalized]
    # Identify extra defects (mentioned in LLM but not detected by YOLO)
    extra = [d for d in llm_defects_normalized if d not in gt_defects_normalized]
    
    return completeness, llm_defects, missing, extra


def evaluate_structural_validation(report, category):
    """
    Validate JSON structure and required fields in LLM report.
    
    This function checks that the LLM-generated report follows the expected JSON structure and contains all required fields. Validates:
    - JSON validity (can be parsed)
    - Presence of required fields (defect_summary, resolution_summary, etc.)
    - Severity fields for appropriate categories
    - Valid severity values (Low/Medium/High)
    - No duplicate defects
    - No redundant fields
    
    Args:
        report: Dictionary containing LLM-generated report
        category: Test case category ('image' or 'text')
    
    Returns:
        Dictionary with validation results:
        - valid_json: Boolean
        - has_required_fields: Boolean
        - has_severity_fields: Boolean
        - valid_severity_values: Boolean
        - no_duplicates: Boolean
        - no_redundant_defects_field: Boolean
        - issues: List of issue descriptions (strings)
    """
    results = {
        'valid_json': True,
        'has_required_fields': True,
        'has_severity_fields': True,
        'valid_severity_values': True,
        'no_duplicates': True,
        'no_redundant_defects_field': True,
        'issues': []
    }
    
    if report is None:
        results['valid_json'] = False
        results['issues'].append("Report missing")
        return results
    
    if 'error' in report:
        results['valid_json'] = False
        results['issues'].append(f"JSON error: {report['error']}")
        return results
    
    # Check for redundant 'defects' field
    if 'defects' in report:
        results['no_redundant_defects_field'] = False
        results['issues'].append("Redundant 'defects' field found")
    
    # Get the actual data (handle both direct and 'sections' structure)
    data = report.get('sections', report)
    
    # Check is_bridge_related
    if data.get('is_bridge_related') == False:
        # Rejection case - should only have is_bridge_related and message
        return results
    
    # Bridge-related case - check required fields
    required_fields = ['is_bridge_related', 'defect_summary', 'resolution_summary', 'further_recommendations']
    
    # For images with YOLO detections, should have detailed_defects
    # For text or images without YOLO, might have overall_severity instead
    has_detailed_defects = 'detailed_defects' in data and len(data.get('detailed_defects', [])) > 0
    has_overall_severity = 'overall_severity' in data
    
    for field in required_fields:
        if field not in data:
            results['has_required_fields'] = False
            results['issues'].append(f"Missing field: {field}")
    
    # If has detailed_defects, validate them
    if has_detailed_defects:
        defects = data['detailed_defects']
        defect_names = []
        
        for i, defect in enumerate(defects):
            # Check required defect fields
            required_defect_fields = ['name', 'engineering_assessment', 'safety_risks', 'recommended_actions', 'severity', 'severity_color']
            for field in required_defect_fields:
                if field not in defect:
                    results['has_severity_fields'] = False
                    results['issues'].append(f"Defect {i+1} missing field: {field}")
            
            # Check severity values
            if 'severity' in defect:
                valid_severities = ['Low', 'Medium', 'High', 'Unknown']
                if defect['severity'] not in valid_severities:
                    results['valid_severity_values'] = False
                    results['issues'].append(f"Invalid severity: {defect['severity']}")
            
            # Check severity colors
            if 'severity_color' in defect:
                valid_colors = ['bg-green-500', 'bg-yellow-500', 'bg-red-500', 'bg-gray-500']
                if defect['severity_color'] not in valid_colors:
                    results['valid_severity_values'] = False
                    results['issues'].append(f"Invalid severity_color: {defect['severity_color']}")
            
            # Track names for duplicate check
            defect_names.append(defect.get('name', '').lower().strip())
        
        # Check for duplicates
        name_counts = Counter(defect_names)
        duplicates = [name for name, count in name_counts.items() if count > 1]
        if duplicates:
            results['no_duplicates'] = False
            results['issues'].append(f"Duplicate defects: {', '.join(duplicates)}")
    
    return results


def run_evaluation():
    """
    Main evaluation function that runs comprehensive assessment of LLM reports.
    
    This function orchestrates the complete evaluation pipeline:
    1. Loads ground truth data
    2. Evaluates each test case across multiple metrics:
       - Classification accuracy (bridge-related vs not)
       - Defect completeness (coverage of YOLO detections)
       - Structural validation (JSON format, required fields)
    3. Calculates aggregate metrics and statistics
    4. Generates visualizations (confusion matrix, completeness plots)
    5. Writes results to CSV and summary files
    
    Output Files:
    - evaluation_results.csv: Per-case detailed results
    - evaluation_summary.txt: Aggregate statistics
    - plots/: Visualization plots (confusion matrix, completeness, etc.)
    
    Returns:
        Exit code 0 on success
    """
    
    print("=" * 80)
    print("LLM Report Evaluation")
    print("=" * 80)
    print()
    
    # Load ground truth
    print("Loading ground truth...")
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} test cases")
    print()
    
    # Create plots directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Results storage
    results = []
    
    # Metrics storage
    classification_metrics = {
        'y_true': [],
        'y_pred': [],
        'correct': 0,
        'total': 0
    }
    
    completeness_metrics = {
        'scores': [],
        'by_defect': defaultdict(lambda: {'total': 0, 'matched': 0})
    }
    
    structural_metrics = {
        'valid_json': 0,
        'has_required_fields': 0,
        'has_severity_fields': 0,
        'valid_severity_values': 0,
        'no_duplicates': 0,
        'no_redundant_defects': 0,
        'total': 0
    }
    
    # Evaluate each test case
    print("Evaluating reports...")
    for test_id, gt in ground_truth.items():
        report = load_report(test_id)
        
        result = {
            'test_id': test_id,
            'category': gt['category'],
            'gt_is_bridge_related': gt['is_bridge_related'],
            'gt_yolo_defects': gt['yolo_detections']
        }
        
        # 1. Classification evaluation
        if gt['is_bridge_related'] is not None:
            pred, is_correct, error = evaluate_classification(report, gt['is_bridge_related'])
            result['pred_is_bridge_related'] = pred
            result['classification_correct'] = is_correct
            result['classification_error'] = error or ""
            
            if pred is not None:
                classification_metrics['y_true'].append(1 if gt['is_bridge_related'] else 0)
                classification_metrics['y_pred'].append(1 if pred else 0)
                classification_metrics['total'] += 1
                if is_correct:
                    classification_metrics['correct'] += 1
        
        # 2. Completeness evaluation (only for images with YOLO detections)
        if gt['category'] == 'image' and gt['yolo_detections_list']:
            completeness, llm_defects, missing, extra = evaluate_defect_completeness(report, gt['yolo_detections_list'])
            result['completeness_score'] = completeness
            result['llm_defects'] = '|'.join(llm_defects) if llm_defects else ""
            result['missing_defects'] = '|'.join(missing) if missing else ""
            result['extra_defects'] = '|'.join(extra) if extra else ""
            
            if completeness is not None:
                completeness_metrics['scores'].append(completeness)
                
                # Per-defect tracking
                for defect in gt['yolo_detections_list']:
                    defect_norm = defect.lower().strip()
                    completeness_metrics['by_defect'][defect_norm]['total'] += 1
                    if defect_norm in [d.lower().strip() for d in llm_defects]:
                        completeness_metrics['by_defect'][defect_norm]['matched'] += 1
        
        # 3. Structural validation
        struct_results = evaluate_structural_validation(report, gt['category'])
        result['valid_json'] = struct_results['valid_json']
        result['has_required_fields'] = struct_results['has_required_fields']
        result['has_severity_fields'] = struct_results['has_severity_fields']
        result['valid_severity_values'] = struct_results['valid_severity_values']
        result['no_duplicates'] = struct_results['no_duplicates']
        result['no_redundant_defects'] = struct_results['no_redundant_defects_field']
        result['structural_issues'] = '; '.join(struct_results['issues']) if struct_results['issues'] else ""
        
        # Update structural metrics
        if report is not None and 'error' not in report:
            structural_metrics['total'] += 1
            for key in ['valid_json', 'has_required_fields', 'has_severity_fields', 
                       'valid_severity_values', 'no_duplicates', 'no_redundant_defects_field']:
                if struct_results[key]:
                    # Map the key correctly
                    if key == 'no_redundant_defects_field':
                        structural_metrics['no_redundant_defects'] += 1
                    else:
                        structural_metrics[key] += 1
        
        results.append(result)
    
    print(f"Evaluated {len(results)} test cases")
    print()
    
    # scikit-learn classification metrics (Accuracy, precision, recall, F1-score) and confusion matrix
    # References:
    #  https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
    #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # ==================================================================
    # Calculate and display metrics
    # ==================================================================
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()
    
    # Classification metrics
    print("1. BINARY CLASSIFICATION (Bridge vs Non-Bridge)")
    print("-" * 80)
    if classification_metrics['total'] > 0:
        accuracy = accuracy_score(classification_metrics['y_true'], classification_metrics['y_pred'])
        precision = precision_score(classification_metrics['y_true'], classification_metrics['y_pred'], zero_division=0)
        recall = recall_score(classification_metrics['y_true'], classification_metrics['y_pred'], zero_division=0)
        f1 = f1_score(classification_metrics['y_true'], classification_metrics['y_pred'], zero_division=0)
        
        print(f"Accuracy:  {accuracy:.4f} ({classification_metrics['correct']}/{classification_metrics['total']})")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(classification_metrics['y_true'], classification_metrics['y_pred'])
        print()
        print("Confusion Matrix:")
        print(f"              Predicted: No  Predicted: Yes")
        print(f"Actual: No    {cm[0][0]:5d}        {cm[0][1]:5d}")
        print(f"Actual: Yes   {cm[1][0]:5d}        {cm[1][1]:5d}")
    else:
        print("No classification data available")
    print()
    
    # Completeness metrics
    print("2. DEFECT COMPLETENESS (YOLO Integration)")
    print("-" * 80)
    if completeness_metrics['scores']:
        avg_completeness = np.mean(completeness_metrics['scores'])
        perfect_count = sum(1 for s in completeness_metrics['scores'] if s == 1.0)
        zero_count = sum(1 for s in completeness_metrics['scores'] if s == 0.0)
        
        print(f"Average Completeness: {avg_completeness:.4f}")
        print(f"Perfect (100%):       {perfect_count}/{len(completeness_metrics['scores'])} ({perfect_count/len(completeness_metrics['scores'])*100:.1f}%)")
        print(f"Zero (0%):            {zero_count}/{len(completeness_metrics['scores'])} ({zero_count/len(completeness_metrics['scores'])*100:.1f}%)")
        print()
        print("Per-Defect Completeness:")
        for defect, stats in sorted(completeness_metrics['by_defect'].items()):
            completeness_pct = (stats['matched'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {defect:25s}: {stats['matched']:3d}/{stats['total']:3d} ({completeness_pct:5.1f}%)")
    else:
        print("No completeness data available (no YOLO detections)")
    print()
    
    # Structural validation
    print("3. STRUCTURAL VALIDATION")
    print("-" * 80)
    if structural_metrics['total'] > 0:
        for key, value in structural_metrics.items():
            if key != 'total':
                pct = (value / structural_metrics['total'] * 100)
                print(f"{key.replace('_', ' ').title():30s}: {value:3d}/{structural_metrics['total']:3d} ({pct:5.1f}%)")
    else:
        print("No structural validation data available")
    print()
    
    # ==================================================================
    # Save results to CSV
    # ==================================================================
    print("Saving results...")
    fieldnames = ['test_id', 'category', 'gt_is_bridge_related', 'gt_yolo_defects', 
                  'pred_is_bridge_related', 'classification_correct', 'classification_error',
                  'completeness_score', 'llm_defects', 'missing_defects', 'extra_defects',
                  'valid_json', 'has_required_fields', 'has_severity_fields', 
                  'valid_severity_values', 'no_duplicates', 'no_redundant_defects', 'structural_issues']
    
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to: {RESULTS_CSV}")
    
    # ==================================================================
    # Generate visualizations
    # ==================================================================
    print("Generating visualizations...")
    
    # 1. Confusion Matrix
    if classification_metrics['total'] > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(classification_metrics['y_true'], classification_metrics['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Not Bridge', 'Bridge'],
                    yticklabels=['Not Bridge', 'Bridge'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix: Bridge vs Non-Bridge Classification')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=300)
        plt.close()
        print(f"Confusion matrix saved")
    
    # 2. Completeness by Defect Type
    if completeness_metrics['by_defect']:
        defect_names = []
        completeness_values = []
        for defect, stats in sorted(completeness_metrics['by_defect'].items()):
            defect_names.append(defect.replace('-', ' ').title())
            completeness_values.append((stats['matched'] / stats['total'] * 100) if stats['total'] > 0 else 0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(defect_names, completeness_values, color='steelblue')
        ax.set_xlabel('Defect Type')
        ax.set_ylabel('Completeness (%)')
        ax.set_title('Defect Completeness by Type')
        ax.set_ylim(0, 105)
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect (100%)')
        ax.legend()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'completeness_by_defect.png'), dpi=300)
        plt.close()
        print(f"Completeness chart saved")
    
    # 3. Severity Distribution (from reports)
    severity_counts = {'Low': 0, 'Medium': 0, 'High': 0, 'Unknown': 0}
    for result in results:
        report = load_report(result['test_id'])
        if report and 'detailed_defects' in report:
            for defect in report['detailed_defects']:
                severity = defect.get('severity', 'Unknown')
                if severity in severity_counts:
                    severity_counts[severity] += 1
    
    if sum(severity_counts.values()) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {'Low': '#22c55e', 'Medium': '#eab308', 'High': '#ef4444', 'Unknown': '#6b7280'}
        bars = ax.bar(severity_counts.keys(), severity_counts.values(),
                     color=[colors[k] for k in severity_counts.keys()])
        ax.set_xlabel('Severity Level')
        ax.set_ylabel('Count')
        ax.set_title('Severity Distribution Across All Defects')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'severity_distribution.png'), dpi=300)
        plt.close()
        print(f"Severity distribution saved")
    
    print()
    print("=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"Results:        {RESULTS_CSV}")
    print(f"Visualizations: {PLOTS_DIR}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_evaluation())