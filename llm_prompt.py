import json
#import subprocess
import pandas as pd
from pathlib import Path
from ollama import generate
from google import genai
from google.genai import types

# now supporting both local and API LLM

#MODEL = "qwen2.5-coder:14b"
MODEL = "gemini-3.5-flash"

API_KEY = "YOUR_API_KEY"
client = genai.Client(api_key=API_KEY)

RAIL_CELLS_PATH = "YOUR_RAILCELL_PATH"
HOTSPOTS_PATH = "YOUR_HOTSPOTS_PATH"

OUTPUT_DIR = Path("OUTPUT_PATH")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_rail_cells(path):
    df = pd.read_csv(path)
    return sorted([
        (int(row), int(col))
        for row, col in df[["row", "col"]].values
    ])



def load_hotspots(path, top_n=80):
    df = pd.read_csv(path)

    hotspots = (
        df.groupby(["x", "y"], as_index=False)["blocked_step_count"]
        .sum()
        .sort_values("blocked_step_count", ascending=False)
        .head(top_n)
    )

    hotspots = hotspots.rename(columns={"x": "row", "y": "col"})

    return [
        {
            "row": int(r["row"]),
            "col": int(r["col"]),
            "blocked_step_count": int(r["blocked_step_count"]),
        }
        for _, r in hotspots.iterrows()
    ]


def build_prompt(rail_cells, hotspots, max_k):
    return f"""
You are designing traffic-light bottlenecks for a Flatland railway environment.

Coordinates are [row, col]. Do not use [x, y].

You may place AT MOST {max_k} traffic-light bottlenecks.
You may use fewer than {max_k} if fewer are justified.

Rules:
- Output valid JSON only.
- Do not write Python code.
- Do not include markdown.
- Every controlled cell must be in rail_cells.
- Every entry cell must be in rail_cells.
- Each bottleneck must have at least 1 controlled cell.
- Each bottleneck must have exactly 2 entry cells: "before" and "after".
- Bottleneck controlled cells should form a compact connected rail segment.
- Bottlenecks must not overlap.
- Prioritize bottlenecks that cover strong blocked-step hotspot clusters.
- Avoid placing traffic lights on random isolated cells.

Your response MUST NOT contain keys named "grid", "blocked_cells", "traffic_lights", or "locations".
The only allowed top-level keys are:
"max_traffic_lights_requested", "num_traffic_lights_returned", "bottlenecks".

Return exactly this JSON schema:

{{
  "max_traffic_lights_requested": {max_k},
  "num_traffic_lights_returned": <integer>,
  "bottlenecks": {{
    "light_1": {{
      "cells": [[row, col], [row, col]],
      "entry_cells": {{
        "before": [row, col],
        "after": [row, col]
      }},
      "reason": "short reason",
      "confidence": "high|medium|low"
    }}
  }}
}}

rail_cells:
{json.dumps(rail_cells)}

hotspots:
{json.dumps(hotspots)}
""".strip()

# depending on whether a local model or an API model is used
# I'd recommend API model due to context size
def call_ollama(prompt):
    response = generate(
        model=MODEL,
        prompt=prompt,
        format="json",
        options={
            "temperature": 0
        },
        stream=False,
    )
    print(response)
    return response["response"].strip()

def call_gemini(prompt):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    print(response)

    return response.text.strip()


def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM output.")

    return json.loads(text[start:end + 1])


def validate_llm_output(config, rail_cells, max_k):
    rail_set = set(rail_cells)

    bottlenecks = config["bottlenecks"]

    assert len(bottlenecks) <= max_k, (
        f"LLM returned {len(bottlenecks)} bottlenecks, max allowed is {max_k}"
    )

    all_controlled = set()

    cleaned = {}

    for name, b in bottlenecks.items():
        cells = {tuple(cell) for cell in b["cells"]}
        entries = {
            key: tuple(value)
            for key, value in b["entry_cells"].items()
        }

        assert len(cells) >= 1, f"{name}: no controlled cells"
        assert set(entries.keys()) == {"before", "after"}, (
            f"{name}: entry_cells must contain exactly before and after"
        )

        missing_cells = cells - rail_set
        missing_entries = set(entries.values()) - rail_set

        assert not missing_cells, f"{name}: controlled cells not on rail: {missing_cells}"
        assert not missing_entries, f"{name}: entry cells not on rail: {missing_entries}"

        overlap = all_controlled & cells
        assert not overlap, f"{name}: overlaps with another bottleneck: {overlap}"

        all_controlled.update(cells)

        cleaned[name] = {
            "cells": sorted([list(c) for c in cells]),
            "entry_cells": {
                "before": list(entries["before"]),
                "after": list(entries["after"]),
            },
            "reason": b.get("reason", ""),
            "confidence": b.get("confidence", "unknown"),
        }

    return {
        "max_traffic_lights_requested": max_k,
        "num_traffic_lights_returned": len(cleaned),
        "bottlenecks": cleaned,
    }


def main(max_k):
    rail_cells = load_rail_cells(RAIL_CELLS_PATH)
    hotspots = load_hotspots(HOTSPOTS_PATH, top_n=80)

    prompt = build_prompt(rail_cells, hotspots, max_k)

    # here again you choose which model to use
    #raw_output = call_ollama(prompt)
    raw_output = call_gemini(prompt)
    parsed = extract_json(raw_output)

    validated = validate_llm_output(parsed, rail_cells, max_k)

    output_path = OUTPUT_DIR / f"traffic_lights_max_k{max_k}.json"

    with open(output_path, "w") as f:
        json.dump(validated, f, indent=2)

    print(f"Saved validated traffic-light config to {output_path}")
    print(json.dumps(validated, indent=2))


if __name__ == "__main__":
    main(max_k=3)
