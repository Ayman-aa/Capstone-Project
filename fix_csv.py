# fix_csv.py
import json
import csv
from pathlib import Path

# Load your raw results
raw_json = Path("experiment_results/raw_results_20260513_191901.json")
with open(raw_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create rows for scoring (anonymized)
rows = []
model_anonymizer = data["model_anonymizer"]

for model_name, model_data in data["results"].items():
    anonymous_name = model_anonymizer[model_name]
    
    for r in model_data["results"]:
        # Clean the response: replace newlines and backticks
        raw_response = r.get("raw_response", "")
        
        # Remove problematic characters for CSV
        clean_for_csv = raw_response.replace('\n', ' ').replace('"', "'").replace('```', '')
        # Truncate if too long
        if len(clean_for_csv) > 500:
            clean_for_csv = clean_for_csv[:500] + "..."
        
        rows.append({
            "model": anonymous_name,
            "prompt_id": r["prompt_id"],
            "category": r["category"],
            "language": r["language"],
            "english_note": r.get("english_note", ""),
            "prompt": r["prompt"],
            "response_preview": clean_for_csv,  # Single line, clean
            "SCORE_0_3": "",
            "NOTES": ""
        })

# Write clean CSV
output_path = Path("experiment_results/blind_scoring_CLEAN.csv")
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["model", "prompt_id", "category", "language", 
                                            "english_note", "prompt", "response_preview", 
                                            "SCORE_0_3", "NOTES"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Saved clean CSV to: {output_path}")
print(f"   Total rows: {len(rows)}")