# pipeline/validator.py

"""
Validator for Raw LLM JSON Output
----------------------------------
This script takes a raw JSON file generated by an LLM (e.g., Gemini),
and performs the following actions:
1.  Validates that essential fields are present (e.g., title, sections).
2.  Generates a permanent, deterministic doc_id from the source filename.
3.  Normalizes date fields into a consistent YYYY-MM-DD format.
4.  Cleans all string fields by trimming extraneous whitespace.
5.  Checks the integrity of nested structures, like the 'sections' array.
6.  Outputs a clean, validated JSON file and a report of its actions.
"""
import os
import json
import hashlib
import argparse
from copy import deepcopy
from typing import Dict, Any, Tuple, List
from dateutil.parser import parse as parse_date, ParserError

# --- Configuration ---
# Define the mandatory fields for a JSON to be considered valid for the next pipeline stage.
REQUIRED_FIELDS: List[str] = ["title", "sections"]

# --- Core Functions ---

def generate_doc_id(filename: str) -> str:
    """Generates a deterministic SHA256 hash for a given filename to use as a document ID."""
    # Using the base name ensures the ID is not dependent on the full path
    base_name = os.path.basename(filename)
    return hashlib.sha256(base_name.encode('utf-8')).hexdigest()

def trim_all_strings(data: Any) -> Any:
    """Recursively traverses a dictionary/list structure and trims whitespace from all strings."""
    if isinstance(data, dict):
        return {k: trim_all_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [trim_all_strings(elem) for elem in data]
    elif isinstance(data, str):
        return data.strip()
    return data

def validate_and_clean_json(data: Dict[str, Any], source_filename: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validates, cleans, and enriches a JSON object from the extraction step.

    Args:
        data (Dict): The raw JSON data loaded from a file.
        source_filename (str): The original filename (e.g., 'my_doc.pdf') to generate a doc_id.

    Returns:
        A tuple containing:
        - The cleaned and validated data dictionary.
        - A validation report dictionary.
    """
    report = {"status": "success", "warnings": [], "errors": []}
    # Create a deep copy to avoid side effects
    cleaned_data = deepcopy(data)

    # 1. Generate and assign a permanent, deterministic doc_id
    cleaned_data["doc_id"] = generate_doc_id(source_filename)

    # 2. Trim all string values for consistency
    cleaned_data = trim_all_strings(cleaned_data)

    # 3. Check for the presence of required fields. This is a critical check.
    for field in REQUIRED_FIELDS:
        if not cleaned_data.get(field):
            report["errors"].append(f"Missing or empty required field: '{field}'")
            report["status"] = "error"
            # Return early if the file is fundamentally invalid for the pipeline
            return cleaned_data, report

    # 4. Attempt to normalize the date format if it exists
    date_str = cleaned_data.get("pbac_meeting_date")
    if date_str:
        try:
            # Intelligently parse the date and format it to YYYY-MM-DD
            parsed_date = parse_date(date_str)
            cleaned_data["pbac_meeting_date"] = parsed_date.strftime("%Y-%m-%d")
        except (ParserError, TypeError):
            report["warnings"].append(f"Could not parse date: '{date_str}'. Setting to null.")
            cleaned_data["pbac_meeting_date"] = None

    # 5. Validate the structure and content of the 'sections' array
    if isinstance(cleaned_data["sections"], list):
        if not cleaned_data["sections"]:
            report["warnings"].append("'sections' array is empty.")
        for i, section in enumerate(cleaned_data["sections"]):
            if not isinstance(section, dict):
                report["warnings"].append(f"Item at index {i} in 'sections' is not a valid object.")
                continue # Skip further checks for this item
            if not section.get("heading"):
                report["warnings"].append(f"Section at index {i} is missing a 'heading'.")
    else:
        report["errors"].append("'sections' field is not a list.")
        report["status"] = "error"


    # Update final status if there were non-critical issues
    if report["warnings"] and report["status"] == "success":
        report["status"] = "success_with_warnings"

    return cleaned_data, report

# --- Main execution block for direct script invocation ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Validate and clean a single extracted JSON file from an LLM.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Path to the raw JSON file to be validated.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the validated JSON file.")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        print(f"📄 Reading raw JSON from: {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # The 'source' field in the JSON holds the original PDF filename, which is best for the doc_id.
        # Fall back to using the input JSON's filename if 'source' is missing.
        source_file_for_id = raw_data.get("source", os.path.basename(args.input))

        # Perform the core validation and cleaning logic
        validated_data, validation_report = validate_and_clean_json(raw_data, source_file_for_id)

        print("\n" + "="*20 + " VALIDATION REPORT " + "="*20)
        print(json.dumps(validation_report, indent=2))
        print("="*59 + "\n")


        # Save the output file ONLY if validation did not produce critical errors
        if validation_report["status"] != "error":
            output_filename = os.path.basename(args.input)
            output_path = os.path.join(args.output_dir, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(validated_data, f, indent=4, ensure_ascii=False)
            print(f"✅ Successfully saved validated JSON to: {output_path}")
        else:
            print("❌ Validation failed with critical errors. Output file not saved.")

    except FileNotFoundError:
        print(f"🔥 ERROR: Input file not found at '{args.input}'")
    except json.JSONDecodeError:
        print(f"🔥 ERROR: Could not parse JSON from '{args.input}'. The file may be corrupt or empty.")
    except Exception as e:
        print(f"🔥 An unexpected error occurred: {e}")
