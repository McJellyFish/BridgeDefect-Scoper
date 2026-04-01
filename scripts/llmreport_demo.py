#!/usr/bin/env python3
"""
LLM Report Generation Module

Author(s): Sai Praneetha, Junjie Ma, I Chen Sung
Date: 12/03/2025
Course: AIT 526 NLP - Natural Language Processing
Project: Multimodal Bridge Defect Scoper
Institution: George Mason University

This module handles bridge defect report generation using Google's Gemini LLM.
It contains the defect knowledge base and prompt engineering for structured
JSON report generation.
"""

import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    error_msg = "Missing GEMINI_API_KEY. Please set it in .env file or environment variable."
    logger.error(error_msg)
    raise ValueError(error_msg)

try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")
    raise

# ---------------------------------------------------------------------
# Knowledge Base for YOLO Labels (Bridge Defects)
# ---------------------------------------------------------------------
DEFECT_KNOWLEDGE_BASE = {
    "bridge-crack": {
        "summary": "Visible fractures or splits in bridge components, indicating tension or structural stress.",
        "engineering_significance": "Cracks are among the most critical indicators of structural weakness. They may expose rebar to corrosion and reduce load-bearing capacity.",
        "common_causes": [
            "Thermal expansion and contraction",
            "Shrinkage of concrete",
            "Reinforcement corrosion-induced expansion",
            "Excessive live load or vibration",
            "Poor construction joints"
        ],
        "safety_impact": "May lead to spalling, delamination, or collapse if ignored.",
        "recommended_actions": [
            "Map and measure crack widths",
            "Conduct structural integrity testing",
            "Seal cracks to prevent moisture ingress",
            "Assess reinforcement corrosion",
            "Schedule urgent repair for active cracks"
        ]
    },

    "for-review": {
        "summary": "Region flagged for human review due to model uncertainty or low confidence.",
        "engineering_significance": "AI detection confidence is low; human validation is required to confirm if it represents a real defect.",
        "common_causes": ["Low image quality", "Obscured or ambiguous defect region"],
        "safety_impact": "Indeterminate — depends on human assessment.",
        "recommended_actions": [
            "Manual visual inspection",
            "Reassess image clarity or angle",
            "If confirmed, reclassify under the appropriate defect type"
        ]
    },

    "material-deterioration": {
        "summary": "Surface or material degradation such as efflorescence, scaling, or erosion.",
        "engineering_significance": "Indicates long-term exposure to moisture, carbonation, or salt intrusion, which weakens concrete surfaces.",
        "common_causes": [
            "Freeze-thaw cycles",
            "Chemical attack or sulfate exposure",
            "Rebar corrosion leading to surface cracking",
            "Water ingress over time"
        ],
        "safety_impact": "May progress to cracking or spalling if untreated.",
        "recommended_actions": [
            "Identify cause (moisture, salt, freeze-thaw)",
            "Clean affected area and remove loose material",
            "Apply protective coating or surface repair",
            "Monitor periodically"
        ]
    },

    "repair": {
        "summary": "Areas showing evidence of patching, sealing, or other repairs.",
        "engineering_significance": "Indicates prior maintenance; quality and durability of repair should be evaluated.",
        "common_causes": [
            "Previous crack sealing or patch repair",
            "Rehabilitation of spalled concrete",
            "Routine maintenance or corrosion mitigation"
        ],
        "safety_impact": "Dependent on repair quality.",
        "recommended_actions": [
            "Assess condition and adhesion of repairs",
            "Check for reoccurring cracks or delamination",
            "Verify that repair materials match structural requirements"
        ]
    },

    "water-leakage": {
        "summary": "Signs of moisture seepage, damp staining, or active water flow through structural components.",
        "engineering_significance": "Persistent leakage accelerates corrosion, efflorescence, and freeze-thaw damage, posing serious risks to long-term durability.",
        "common_causes": [
            "Failed waterproofing membranes",
            "Joint sealant degradation",
            "Blocked or damaged drainage systems",
            "Cracks allowing water ingress"
        ],
        "safety_impact": "Continuous water ingress can lead to rebar corrosion and structural weakening.",
        "recommended_actions": [
            "Identify and eliminate the source of leakage",
            "Repair joints and reapply waterproofing membranes",
            "Inspect drainage and weep holes",
            "Test for chloride or moisture penetration"
        ]
    }
}

# ---------------------------------------------------------------------
# Structured LLM Report Generation
# ---------------------------------------------------------------------
# Google Gemini API usage patterns and multimodal input handling for report generation
# References:
#  https://ai.google.dev/gemini-api/docs/


def generate_bridge_report(defect_label: str, image_path: str = None, detected_labels: list = None) -> str:
    """
    Generate a structured bridge defect report using Google's Gemini LLM.

    This function is the core of the LLM integration, creating comprehensive
    engineering reports based on YOLO detections or text descriptions. It
    handles three distinct cases with different prompt strategies:

    1. YOLO-detected defects: Uses knowledge base context to generate detailed defect-specific analysis for each detected type

    2. Text-only input: Analyzes text description and provides overall assessmentwith severity rating

    3. Image-only (no detections): Validates if image is bridge-related and provides general analysis

    The function uses prompt engineering to ensure consistent JSON output format and is suitable for web UI display.

    Args:
        defect_label: Summary string describing defects or text input description
        image_path: Optional path to image file for multimodal analysis
        detected_labels: Optional list of defect names detected by YOLO

    Returns:
        String containing JSON-formatted report with:
        - is_bridge_related: Boolean validation
        - defect_summary: Overall condition summary
        - resolution_summary: Recommended actions
        - detailed_defects: List of defect-specific assessments (if YOLO detections)
        - overall_severity: Severity rating (for text-only inputs)
        - further_recommendations: Additional engineering recommendations

    Raises:
        Various exceptions are caught and returned as error strings starting with "(Gemini error:" for handling by the calling function.
    """
    try:
        # Initialize Gemini model (version 2.5 Flash - optimized for speed)
        model = genai.GenerativeModel("gemini-2.5-flash")

        # Normalize detected_labels parameter
        # Use provided detected_labels or default to empty list
        if detected_labels is None:
            detected_labels = []

        # Remove duplicates and ensure it's a list (in case set or tuple passed)
        detected_labels = list(set(detected_labels))

        print("Detected YOLO labels:", detected_labels)  # debug log

        # Fallback: Try to infer defect types from label string if YOLO found nothing
        # This helps in edge cases where YOLO didn't detect but defect_label mentions types
        if not detected_labels and image_path is not None:
            detected_labels = [
                key for key in DEFECT_KNOWLEDGE_BASE.keys()
                if key in defect_label.lower()
            ]

        # ---------------------------------------------------------------------
        # Case 1: Known labeled defects (YOLO detected defects)
        # ---------------------------------------------------------------------
        # When YOLO has detected specific defects, use knowledge base context to generate detailed, defect-specific engineering assessments.
        if detected_labels:
            # Build knowledge base context from detected defects. Each defect type has engineering information (causes, risks, actions)
            # This context guides the LLM to generate accurate, domain-specific reports
            sections = []
            for label in detected_labels:
                kb = DEFECT_KNOWLEDGE_BASE[label]
                causes_str = '\n- '.join(kb['common_causes'])
                actions_str = '\n- '.join(kb['recommended_actions'])

                label_block = f"""
                #### {label.replace('-', ' ').title()}

                **Summary:** {kb['summary']}

                **Engineering Significance:** {kb['engineering_significance']}

                **Common Causes:**
                - {causes_str}

                **Safety Impact:** {kb['safety_impact']}

                **Recommended Actions:**
                - {actions_str}
                """
                sections.append(label_block)
            # Combine all defect knowledge into single context string
            kb_context = "\n\n".join(sections)

            # Construct prompt with knowledge base context for detected defects
            # The prompt instructs LLM to act as expert engineer and use provided knowledge
            prompt = f"""
            You are a senior bridge inspection engineer. Generate a professional report for the detected defects: {', '.join(detected_labels)}.

            Use the following knowledge base to generate a JSON-structured report.

            ---
            ### REQUIRED JSON STRUCTURE
            Output valid JSON with this exact structure:
            {{
              "is_bridge_related": true,
              "defect_summary": "string - 2-4 lines summarizing all defects concisely",
              "resolution_summary": "string - 2-4 sentences on key corrective actions",
              "detailed_defects": [
                {{
                  "name": "string - defect name",
                  "engineering_assessment": "string - causes, mechanisms, impact",
                  "safety_risks": "string - potential hazards",
                  "recommended_actions": "string - repair/inspection steps"
                }},
                ... for ALL defects including for-review/ Include each defect only once.
              ],
              "further_recommendations": "string - additional recommendations"
            }}

            Note: For "for-review" defects, explain that AI confidence is low and manual inspection is required to confirm the defect type. Safety risks should note that severity is indeterminate pending human assessment.

            Output ONLY the JSON object, nothing else.
            ---

            Knowledge Base Context:
            {kb_context}
            """

        # ---------------------------------------------------------------------
        # Case 2: No detected labels — general mode
        # ---------------------------------------------------------------------
        # When YOLO didn't detect defects (or no image provided), use general analysis
        else:
            if image_path is None:
                # Text-only input: User provides defect description in natural language
                # Assume bridge-related (validation happens in app_demo.py)
                prompt = f"""
                You are a senior bridge inspection engineer. Analyze this bridge defect description based on the given text: {defect_label}

                Ensure to assess the overall severity of the described bridge condition and provide a comprehensive report.

                ---
                ### REQUIRED JSON STRUCTURE
                Output valid JSON with this exact structure (DO NOT include detailed_defects):
                {{
                  "is_bridge_related": true,
                  "defect_summary": "string - 2-4 lines describing overall condition",
                  "resolution_summary": "string - 2-4 sentences on inspection/maintenance recommendations",
                  "further_recommendations": "string - additional recommendations",
                  "overall_severity": "Low|Medium|High - REQUIRED: Assess severity as Low for minor issues, Medium for moderate damage, High for critical safety concerns"
                }}

                Examples of severity assessment:
                - Low: "Minor surface cracking, no structural impact"
                - Medium: "Moderate corrosion, requires monitoring"
                - High: "Severe cracking with exposed rebar, immediate repair needed"

                Output ONLY the JSON object, nothing else.
                ---
                """
            else:
                # Image-only input with no YOLO detections
                # LLM validates if image is bridge-related and provides general analysis
                prompt = f"""
                You are a senior bridge inspection engineer. Determine if this image is related to bridge defects or bridge inspection.

                Output a JSON object with "is_bridge_related": true or false.
                If is_bridge_related is false, include only "message": "Please provide input related to bridge defects or inspection."
                If is_bridge_related is true, include the full report structure.

                ---
                ### REQUIRED JSON STRUCTURE
                {{
                  "is_bridge_related": true,
                  "defect_summary": "string - 2-4 lines describing overall condition",
                  "resolution_summary": "string - 2-4 sentences on inspection/maintenance recommendations",
                  "detailed_defects": [],
                  "further_recommendations": "string - additional recommendations",
                  "overall_severity": "Low|Medium|High - assessed severity based on description"
                }}

                Output ONLY the JSON object, nothing else.
                ---
                """

        # ---------------------------------------------------------------------
        # Run Gemini API call (multimodal if image provided)
        # ---------------------------------------------------------------------
        # Gemini supports both text-only and multimodal (text + image) requests
        # Log first 500 chars of prompt
        print(f"LLM LOG: Sending prompt to Gemini: {prompt[:500]}...")

        if image_path and os.path.exists(image_path):
            # Multimodal request: Send image + text prompt
            # Gemini can analyze image content along with text instructions
            print(
                f"LLM LOG: Sending multimodal request with image: {image_path}")
            response = model.generate_content([
                genai.upload_file(image_path),  # Upload image for analysis
                prompt  # Text prompt with instructions
            ])
        else:
            # Text-only request: No image, just prompt
            print("LLM LOG: Sending text-only request")
            response = model.generate_content(prompt)

        llm_response = response.text.strip()
        print(f"LLM LOG: Received response: {repr(llm_response)}")

        return llm_response

    except Exception as e:
        logger.error(
            f"Error generating bridge report: {str(e)}", exc_info=True)
        return f"(Gemini error: {e})"
