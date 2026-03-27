#!/usr/bin/env python3
"""
DeepEval LLM Report Evaluation

Author(s): Sai Praneetha, Junjie Ma, I Chen Sung
Date: 12/03/2025
Course: AIT 526 NLP - Natural Language Processing
Project: Multimodal Bridge Defect Scoper
Institution: George Mason University

Uses OpenAI GPT as judge to evaluate Gemini-generated bridge inspection reports.

This version:
- Uses ground_truth.csv only to select test_ids and optionally provide YOLO defects as soft context.
- Does NOT require the report to list all YOLO defects.
- Uses a generic rubric-style expected_output so the LLM judge evaluates overall bridge-defect report quality.
- Prints summary to console and saves plots under evaluation_data/ground_truth/results/deepeval_plots/.
"""

import os
import json
import sys
import csv
from pathlib import Path
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    HallucinationMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from typing import List, Dict, Any
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Paths - using centralized config
# ---------------------------------------------------------------------
from config import (
    GROUND_TRUTH_CSV as GROUND_TRUTH_PATH,
    CASE1_PREDICTION_REPORTS as CASE1_REPORTS,
    CASE2_PREDICTION_REPORTS as CASE2_REPORTS,
    DEEPEVAL_PLOTS_DIR as PLOTS_DIR,
    PROJECT_ROOT
)

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Try both possible env variable names
OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPEN_AI_API_KEY or OPENAI_API_KEY not found in .env file")
    sys.exit(1)

# Set for OpenAI SDK
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Model configuration
MODEL_NAME = "gpt-4o-mini"

print(f"Using OpenAI model as judge: {MODEL_NAME}")


# ---------------------------------------------------------------------
# Helpers: loading ground truth and reports
# ---------------------------------------------------------------------
def load_ground_truth() -> Dict[str, Dict[str, Any]]:
    """
    Load ground truth CSV file for metadata reference.
    
    In DeepEval evaluation, ground truth is used primarily for metadata (test case IDs, YOLO detections context) rather than strict validation.
    The LLM-as-Judge approach evaluates report quality holistically rather than exact matching.
    
    Returns:
        Dictionary mapping test_id to ground truth metadata dictionary.
    """
    ground_truth: Dict[str, Dict[str, Any]] = {}
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_id = row["test_id"]

            # Convert string booleans
            if row["is_bridge_related"] == "True":
                row["is_bridge_related"] = True
            elif row["is_bridge_related"] == "False":
                row["is_bridge_related"] = False
            else:
                row["is_bridge_related"] = None

            # YOLO detections list
            if row["yolo_detections"] and row["yolo_detections"] != "N/A":
                row["yolo_detections_list"] = row["yolo_detections"].split("|")
            else:
                row["yolo_detections_list"] = []

            ground_truth[test_id] = row

    return ground_truth


def load_report(test_id: str) -> Any:
    """
    Load JSON prediction report for a given test case ID.
    
    Args:
        test_id: Test case identifier (e.g., "case1_1", "case2_01")
    
    Returns:
        Dictionary containing parsed JSON report, None if not found,
        or error dictionary if parsing fails.
    """
    if test_id.startswith("case1_"):
        img_num = test_id.split("_")[1]
        report_path = CASE1_REPORTS / f"{img_num}.json"
    elif test_id.startswith("case2_"):
        text_num = test_id.split("_")[1]
        report_path = CASE2_REPORTS / f"{text_num}.json"
    else:
        return None

    if not report_path.exists():
        return None

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def extract_report_text(report: Any) -> str:
    """
    Extract readable text from JSON report for DeepEval evaluation.
    
    Converts structured JSON report into plain text format that DeepEval's LLM-as-Judge can evaluate.
    This allows the judge LLM to assess report quality, completeness, and relevance in natural language.
    
    Args:
        report: Dictionary containing LLM-generated report JSON
    
    Returns:
        Plain text string summarizing the report content, or empty string if the report is invalid or has errors.
    """
    if report is None or "error" in report:
        return ""

    # Handle rejection/error cases (valid=false or rejection messages)
    if report.get("valid") is False:
        return report.get("message", "Report validation failed")
    
    if report.get("is_bridge_related") is False:
        return report.get("message", "Input not related to bridge defects")

    # Handle both direct root and 'sections' structure
    data = report.get("sections", report)

    parts: List[str] = []

    if "defect_summary" in data:
        parts.append(f"Summary: {data['defect_summary']}")

    if "detailed_defects" in data and data["detailed_defects"]:
        parts.append("\nDefects:")
        for defect in data["detailed_defects"]:
            name = defect.get("name", "Unknown defect")
            parts.append(f"- {name}")
            if "engineering_assessment" in defect:
                parts.append(f"  Assessment: {defect['engineering_assessment']}")
            if "safety_risks" in defect:
                parts.append(f"  Risks: {defect['safety_risks']}")
            if "recommended_actions" in defect:
                parts.append(f"  Actions: {defect['recommended_actions']}")

    if "resolution_summary" in data:
        parts.append(f"\nResolution: {data['resolution_summary']}")

    if "overall_severity" in data:
        parts.append(f"\nOverall Severity: {data['overall_severity']}")

    result = "\n".join(parts).strip()
    
    # If we still have nothing, check for any text in the report
    if not result:
        # Try to extract any meaningful text from the report
        if "message" in report:
            result = report["message"]
        elif "defect_summary" in report:
            result = report["defect_summary"]
    
    return result


# ---------------------------------------------------------------------
# Test case creation: softer, rubric-based expected_output
# ---------------------------------------------------------------------
# DeepEval LLMTestCase creation and framework usage
# References:
#  https://docs.confident-ai.com/docs/
def create_test_cases(ground_truth: Dict[str, Dict[str, Any]] = None, limit: int = None):
    """
    Create DeepEval test cases by scanning all reports in predictions directories.
    
    This function builds test cases for LLM-as-Judge evaluation using DeepEval.
    It processes all generated reports and creates test cases with:
    - Input: Original image/text input or YOLO detections
    - Output: LLM-generated report text
    - Expected Output: Generic rubric describing good bridge defect reports
    - Context: YOLO detections and ground truth metadata
    
    The LLM-as-Judge approach uses GPT-4o-mini to evaluate report quality using metrics like faithfulness, relevancy, and hallucination detection.
    
    Args:
        ground_truth: Optional ground truth dictionary for metadata context
        limit: Optional limit on number of test cases to create (for testing)
    
    Returns:
        Tuple of (test_cases_list, metadata_list) for DeepEval evaluation
    """
    test_cases: List[LLMTestCase] = []
    test_case_metadata: List[Dict[str, Any]] = []
    
    if ground_truth is None:
        ground_truth = {}

    # Scan Case 1: Images
    print("   Scanning evaluation_data/predictions/case1/reports...")
    case1_files_raw = list(CASE1_REPORTS.glob("*.json"))
    # Sort numerically (e.g., 1.json, 2.json, ..., 10.json, not 1.json, 10.json, 2.json)
    case1_files = sorted(case1_files_raw, key=lambda f: int(f.stem) if f.stem.isdigit() else 999999)
    print(f"      Found {len(case1_files)} Case 1 reports")
    
    for report_path in case1_files:
        img_num = report_path.stem  # e.g., "1" from "1.json"
        test_id = f"case1_{img_num}"
        
        # Load report
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
        except Exception as e:
            print(f"      Warning: Could not load {test_id}: {e}")
            continue
        
        if "error" in report:
            continue
        
        actual_output = extract_report_text(report)
        if not actual_output:
            continue
        
        # Get metadata from ground truth if available
        gt = ground_truth.get(test_id, {})
        yolo_detections_list = gt.get("yolo_detections_list", [])
        
        # Soft context from YOLO detections (if available)
        context: List[str] = []
        if yolo_detections_list:
            context.append(
                "Possible defects detected by the vision model: "
                + ", ".join(yolo_detections_list)
            )
        else:
            context.append(
                "No specific defects detected by the vision model, or detection data not available."
            )
        
        # Generic knowledge context
        retrieval_context = [
            "Bridge inspection reports should focus on structural condition, visible defects, and safety implications.",
            "Common bridge defects include: cracks, water leakage, corrosion, material deterioration, and joint or bearing issues.",
            "A good report explains severity, likely causes, and recommended inspection or repair actions.",
            "For non-bridge inputs, the system should correctly reject them with an appropriate message.",
        ]
        
        # Determine expected output based on report type
        is_bridge = report.get("is_bridge_related", True)
        if report.get("valid") is False or is_bridge is False:
            # This is a rejection/error case
            expected_output = (
                "If the input is not bridge-related, the system should correctly identify this "
                "and return an appropriate rejection message asking for bridge-related input."
            )
        else:
            # This is a valid bridge report
            expected_output = (
                "A high-quality bridge defect inspection report that:\n"
                "- Clearly states that the structure is a bridge or bridge component.\n"
                "- Describes any visible or likely defects (e.g. cracks, leakage, deterioration) in a technically plausible way.\n"
                "- Explains engineering significance and safety implications in simple yet accurate language.\n"
                "- Provides concrete, actionable recommendations for inspection, monitoring, or repair.\n"
            )
        
        test_case = LLMTestCase(
            input=f"Assess the quality of this bridge defect report for test ID: {test_id}",
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
        )
        
        metadata = {
            "test_id": test_id,
            "category": gt.get("category", "image"),
            "yolo_defects": gt.get("yolo_detections", "N/A"),
            "yolo_count": len(yolo_detections_list),
            "is_bridge_related": is_bridge if is_bridge is not None else gt.get("is_bridge_related", None),
        }
        
        test_cases.append(test_case)
        test_case_metadata.append(metadata)
    
    # Scan Case 2: Text inputs
    print("   Scanning evaluation_data/predictions/case2/reports...")
    case2_files_raw = list(CASE2_REPORTS.glob("*.json"))
    # Sort numerically (e.g., 01.json, 02.json, ..., 100.json, not 01.json, 100.json, 02.json)
    case2_files = sorted(case2_files_raw, key=lambda f: int(f.stem) if f.stem.isdigit() else 999999)
    print(f"      Found {len(case2_files)} Case 2 reports")
    
    for report_path in case2_files:
        text_num = report_path.stem  # e.g., "01" from "01.json" or "146" from "146.json"
        test_id = f"case2_{int(text_num):02d}"  # Convert to int then format with leading zero
        
        # Load report
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
        except Exception as e:
            print(f"      Warning: Could not load {test_id}: {e}")
            continue
        
        if "error" in report:
            continue
        
        actual_output = extract_report_text(report)
        if not actual_output:
            continue
        
        # Get metadata from ground truth if available
        gt = ground_truth.get(test_id, {})
        
        # Text inputs don't have YOLO detections
        context: List[str] = [
            "This is a text-only input. No visual defects were detected.",
        ]
        
        # Generic knowledge context
        retrieval_context = [
            "Bridge inspection reports should focus on structural condition, visible defects, and safety implications.",
            "Common bridge defects include: cracks, water leakage, corrosion, material deterioration, and joint or bearing issues.",
            "A good report explains severity, likely causes, and recommended inspection or repair actions.",
            "For non-bridge inputs, the system should correctly reject them with an appropriate message.",
        ]
        
        # Determine expected output based on report type
        is_bridge = report.get("is_bridge_related", True)
        if report.get("valid") is False or is_bridge is False:
            # This is a rejection/error case
            expected_output = (
                "If the input is not bridge-related, the system should correctly identify this "
                "and return an appropriate rejection message asking for bridge-related input."
            )
        else:
            # This is a valid bridge report (text input)
            expected_output = (
                "A high-quality bridge defect inspection report based on text description that:\n"
                "- Clearly identifies the structure as bridge-related.\n"
                "- Describes defects or conditions mentioned in the input in a technically plausible way.\n"
                "- Explains engineering significance and safety implications.\n"
                "- Provides concrete, actionable recommendations.\n"
            )
        
        test_case = LLMTestCase(
            input=f"Assess the quality of this bridge defect report for test ID: {test_id}",
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
        )
        
        metadata = {
            "test_id": test_id,
            "category": gt.get("category", "text"),
            "yolo_defects": "N/A",
            "yolo_count": 0,
            "is_bridge_related": is_bridge if is_bridge is not None else gt.get("is_bridge_related", None),
        }
        
        test_cases.append(test_case)
        test_case_metadata.append(metadata)
    
    # Apply limit if specified
    if limit and len(test_cases) > limit:
        test_cases = test_cases[:limit]
        test_case_metadata = test_case_metadata[:limit]
        print(f"   [LIMIT] Restricted to {limit} test cases")
    
    return test_cases, test_case_metadata


# ---------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------
# DeepEval evaluation framework and metrics (Faithfulness, Answer Relevancy, Hallucination):
# References:
#  https://docs.confident-ai.com/docs/get-started
#  https://docs.confident-ai.com/docs/metrics/overview
def run_evaluation() -> int:
    """
    Main evaluation function using DeepEval's LLM-as-Judge approach.
    
    This function orchestrates comprehensive evaluation using OpenAI GPT-4o-mini as a judge to evaluate Gemini-generated reports.
    
    It uses multiple metrics:
    - Faithfulness: Does the report stick to YOLO detections and knowledge base?
    - Answer Relevancy: Is the report relevant to detected defects?
    - Hallucination: Does the LLM invent defects not detected by YOLO?
    - G-Eval: Holistic quality score (1-5 scale)
    
    The evaluation processes all reports in the predictions directory and generates detailed metric scores, summary statistics, and visualization plots.
    
    Output Files:
    - deepeval_results.csv: Detailed metric scores per test case
    - deepeval_summary.txt: Aggregate statistics
    - deepeval_plots/: Visualization plots (distributions, correlations, etc.)
    
    Returns:
        Exit code 0 on success, 1 on error
    """
    print("=" * 80)
    print("DeepEval LLM Report Evaluation")
    print(f"Judge Model: {MODEL_NAME}")
    print("=" * 80)
    print()

    # Load ground truth metadata
    print("Loading ground truth metadata...")
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} entries from ground_truth.csv")

    # Create test cases - now evaluates ALL reports in evaluation_data/predictions directories
    print("Creating DeepEval test cases (all reports in evaluation_data/predictions)...")
    test_cases, test_case_metadata = create_test_cases(ground_truth)
    print(f"Created {len(test_cases)} test cases for evaluation")
    print()

    if len(test_cases) == 0:
        print("No valid test cases to evaluate!")
        return 1

    # Optional: small test mode
    TEST_MODE = os.getenv("DEEPEVAL_TEST_MODE", "false").lower() == "true"
    if TEST_MODE and len(test_cases) > 3:
        test_cases, test_case_metadata = test_cases[:3], test_case_metadata[:3]
        print(f"[TEST MODE] Limiting to {len(test_cases)} test cases for debugging")
        print("[NOTE] Unset DEEPEVAL_TEST_MODE to run full evaluation")
        print()

    # Initialize metrics (rubric-style)
    print("Initializing evaluation metrics...")
    metrics = [
        FaithfulnessMetric(
            model=MODEL_NAME,
            threshold=0.7,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            model=MODEL_NAME,
            threshold=0.7,
            include_reason=True,
        ),
        HallucinationMetric(
            model=MODEL_NAME,
            threshold=0.5,
            include_reason=True,
        ),
        GEval(
            name="Bridge_Report_Quality",
            model=MODEL_NAME,
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            evaluation_steps=[
                "Check if the report clearly addresses a bridge structure and its condition.",
                "Check if the report meaningfully discusses defects or potential structural issues.",
                "Check if the reasoning is technically plausible and non-contradictory.",
                "Check if recommendations are concrete and appropriate for inspection/repair.",
            ],
            threshold=3.5,  # On a 1–5 style internal scale
        ),
    ]

    print(f"Initialized {len(metrics)} metrics:")
    for m in metrics:
        print(f"-{m.__class__.__name__}")
    print()

    # Run evaluation in batches (to be gentle with rate limits)
    print("Running evaluation...")
    import time

    batch_size = 20
    num_batches = (len(test_cases) + batch_size - 1) // batch_size

    # {global_idx: {metric_name: score}}
    test_case_scores: Dict[int, Dict[str, float]] = {}

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(test_cases))
        batch = test_cases[start_idx:end_idx]

        print(
            f"Batch {batch_num + 1}/{num_batches}: "
            f"evaluating test cases {start_idx + 1}-{end_idx}..."
        )

        try:
            # Fresh metric instances per batch (safer with some implementations)
            batch_metrics = [
                FaithfulnessMetric(
                    model=MODEL_NAME,
                    threshold=0.7,
                    include_reason=True,
                ),
                AnswerRelevancyMetric(
                    model=MODEL_NAME,
                    threshold=0.7,
                    include_reason=True,
                ),
                HallucinationMetric(
                    model=MODEL_NAME,
                    threshold=0.5,
                    include_reason=True,
                ),
                GEval(
                    name="Bridge_Report_Quality",
                    model=MODEL_NAME,
                    evaluation_params=[
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                        LLMTestCaseParams.EXPECTED_OUTPUT,
                    ],
                    evaluation_steps=[
                        "Check if the report clearly addresses a bridge structure and its condition.",
                        "Check if the report meaningfully discusses defects or potential structural issues.",
                        "Check if the reasoning is technically plausible and non-contradictory.",
                        "Check if recommendations are concrete and appropriate for inspection/repair.",
                    ],
                    threshold=3.5,
                ),
            ]

            batch_eval_result = evaluate(
                test_cases=batch,
                metrics=batch_metrics,
            )

            # Extract scores back into test_case_scores
            # DeepEval returns an EvaluationResult with test_results
            if hasattr(batch_eval_result, "test_results") and batch_eval_result.test_results:
                results_list = batch_eval_result.test_results
            else:
                # Fallback: treat batch_eval_result as list-like
                results_list = getattr(batch_eval_result, "results", []) or []

            for i, test_result in enumerate(results_list):
                global_idx = start_idx + i
                if global_idx not in test_case_scores:
                    test_case_scores[global_idx] = {}

                # Primary: metrics_data dict or list
                metrics_data = getattr(test_result, "metrics_data", None)
                if metrics_data:
                    if isinstance(metrics_data, dict):
                        items = metrics_data.items()
                    else:
                        items = [(getattr(m, "name", ""), m) for m in metrics_data]

                    for key, metric_obj in items:
                        mname = (
                            key.lower()
                            if isinstance(key, str)
                            else getattr(metric_obj, "name", "").lower()
                        )

                        score = getattr(metric_obj, "score", None)
                        if score is None and hasattr(metric_obj, "verdict"):
                            verdict = getattr(metric_obj, "verdict")
                            if isinstance(verdict, (int, float)):
                                score = float(verdict)

                        if score is not None:
                            score = float(score)
                            if "faithfulness" in mname:
                                test_case_scores[global_idx]["faithfulness"] = score
                            elif "relevancy" in mname:
                                test_case_scores[global_idx]["answer_relevancy"] = score
                            elif "hallucination" in mname:
                                test_case_scores[global_idx]["hallucination"] = score
                            elif (
                                "geval" in mname
                                or "report" in mname
                                or "quality" in mname
                            ):
                                # Some GEval variants may be 0-1; some 1-5
                                if score <= 1.0:
                                    test_case_scores[global_idx]["g_eval"] = score * 5.0
                                else:
                                    test_case_scores[global_idx]["g_eval"] = score

            if batch_num < num_batches - 1:
                time.sleep(8)

        except Exception as e:
            print(f"Batch {batch_num + 1} failed: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    print()
    print(
        f"Processed {len(test_case_scores)}/{len(test_cases)} "
        f"test cases with metric scores"
    )
    print()

    # -----------------------------------------------------------------
    # Aggregate results (for console summary + plots)
    # -----------------------------------------------------------------
    metric_scores: Dict[str, List[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "hallucination": [],
        "g_eval": [],
    }
    detailed_results: List[Dict[str, Any]] = []

    for i, meta in enumerate(test_case_metadata):
        result = {
            "test_id": meta["test_id"],
            "category": meta["category"],
            "yolo_defects": meta["yolo_defects"],
            "yolo_count": meta["yolo_count"],
        }

        scores = test_case_scores.get(i, {})
        f_score = scores.get("faithfulness")
        a_score = scores.get("answer_relevancy")
        h_score = scores.get("hallucination")
        g_score = scores.get("g_eval")

        if f_score is not None:
            metric_scores["faithfulness"].append(f_score)
        if a_score is not None:
            metric_scores["answer_relevancy"].append(a_score)
        if h_score is not None:
            metric_scores["hallucination"].append(h_score)
        if g_score is not None:
            metric_scores["g_eval"].append(g_score)

        result["faithfulness"] = f_score
        result["answer_relevancy"] = a_score
        result["hallucination"] = h_score
        result["g_eval"] = g_score

        detailed_results.append(result)

    # -----------------------------------------------------------------
    # Console summary (no CSV, just print + plots)
    # -----------------------------------------------------------------
    print("=" * 80)
    print("EVALUATION RESULTS (summary)")
    print("=" * 80)
    print()

    print(f"Total Test Cases:          {len(test_cases)}")
    print(f"Successfully Scored Cases: {len(test_case_scores)}")
    print(f"Failed / missing scores:   {len(test_cases) - len(test_case_scores)}")
    print()

    for metric_name, scores in metric_scores.items():
        if scores:
            avg = np.mean(scores)
            std = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)

            display_name = metric_name.replace("_", " ").title()
            print(
                f"{display_name:20s}: "
                f"{avg:.3f} ± {std:.3f}  (min: {min_score:.3f}, max: {max_score:.3f})"
            )

    if metric_scores["hallucination"]:
        halluc_count = sum(1 for s in metric_scores["hallucination"] if s > 0.5)
        rate = halluc_count / len(metric_scores["hallucination"]) * 100.0
        print(f"\nHallucination Rate (>0.5): {rate:.1f}%")

    print()

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    print("Generating plots...")
    PLOTS_DIR.mkdir(exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    # 1) Metric distributions
    if (
        metric_scores["faithfulness"]
        or metric_scores["answer_relevancy"]
        or metric_scores["hallucination"]
    ):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        if metric_scores["faithfulness"]:
            axes[0].hist(
                metric_scores["faithfulness"],
                bins=20,
                edgecolor="black",
            )
            axes[0].set_xlabel("Faithfulness Score")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Faithfulness Distribution")
            axes[0].axvline(
                np.mean(metric_scores["faithfulness"]),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(metric_scores['faithfulness']):.3f}",
            )
            axes[0].legend()

        if metric_scores["answer_relevancy"]:
            axes[1].hist(
                metric_scores["answer_relevancy"],
                bins=20,
                edgecolor="black",
            )
            axes[1].set_xlabel("Answer Relevancy Score")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Answer Relevancy Distribution")
            axes[1].axvline(
                np.mean(metric_scores["answer_relevancy"]),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(metric_scores['answer_relevancy']):.3f}",
            )
            axes[1].legend()

        if metric_scores["hallucination"]:
            axes[2].hist(
                metric_scores["hallucination"],
                bins=20,
                edgecolor="black",
            )
            axes[2].set_xlabel("Hallucination Score")
            axes[2].set_ylabel("Count")
            axes[2].set_title("Hallucination Distribution")
            axes[2].axvline(
                np.mean(metric_scores["hallucination"]),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(metric_scores['hallucination']):.3f}",
            )
            axes[2].legend()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "metric_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f" Saved: {PLOTS_DIR / 'metric_distributions.png'}")

    # 2) Metric comparison bar chart
    metric_names = []
    metric_means = []
    metric_stds = []

    for key, label in [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("hallucination", "Hallucination"),
        ("g_eval", "G-Eval (0–5)"),
    ]:
        if metric_scores[key]:
            metric_names.append(label)
            metric_means.append(float(np.mean(metric_scores[key])))
            metric_stds.append(float(np.std(metric_scores[key])))

    if metric_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            metric_names,
            metric_means,
            yerr=metric_stds,
            capsize=5,
            alpha=0.8,
            edgecolor="black",
        )
        ax.set_ylabel("Score")
        ax.set_title("Average Metric Scores (with Std Dev)")

        max_score = max(metric_means) + (max(metric_stds) if metric_stds else 0)
        ylim_max = max(1.1, max_score * 1.1) if max_score < 5 else 5.5
        ax.set_ylim(0, ylim_max)

        for bar, mean, std in zip(bars, metric_means, metric_stds):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{mean:.3f}±{std:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "metric_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f" Saved: {PLOTS_DIR / 'metric_comparison.png'}")

    # 3) G-Eval distribution
    if metric_scores["g_eval"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(
            metric_scores["g_eval"], bins=15, edgecolor="black", alpha=0.8
        )
        ax.set_xlabel("G-Eval Score (0–5)")
        ax.set_ylabel("Count")
        ax.set_title("G-Eval Quality Distribution")
        ax.axvline(
            np.mean(metric_scores["g_eval"]),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(metric_scores['g_eval']):.2f}",
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "g_eval_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f" Saved: {PLOTS_DIR / 'g_eval_distribution.png'}")

    # 4) Hallucination rate pie
    if metric_scores["hallucination"]:
        halluc_count = sum(1 for s in metric_scores["hallucination"] if s > 0.5)
        clean_count = len(metric_scores["hallucination"]) - halluc_count

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            [halluc_count, clean_count],
            labels=[
                f"With Hallucinations ({halluc_count})",
                f"Clean ({clean_count})",
            ],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Hallucination Rate (>0.5 threshold)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "hallucination_rate.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f" Saved: {PLOTS_DIR / 'hallucination_rate.png'}")

    # 5) Optional: correlation heatmap and score vs YOLO count
    if detailed_results and len(detailed_results) > 5:
        df = pd.DataFrame(detailed_results)
        numeric_cols = ["faithfulness", "answer_relevancy", "hallucination", "g_eval", "yolo_count"]
        df_numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()

        if len(df_numeric) > 3:
            corr = df_numeric.corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr,
                annot=True,
                fmt=".3f",
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                ax=ax,
            )
            ax.set_title("Metric Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f" Saved: {PLOTS_DIR / 'correlation_heatmap.png'}")

    print()
    print("=" * 80)
    print("DeepEval evaluation complete.")
    print(f"Plots saved under: {PLOTS_DIR}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_evaluation())
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)