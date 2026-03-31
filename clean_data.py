import os
import glob
import json
import argparse
import shutil
import numpy as np
import nltk
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm
import re
from copy import deepcopy
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

VALID_CATEGORY_IDS = set(range(1, 10)) 
tokenizer = TreebankWordTokenizer() #Move outside functions to avoid re-initialization


def parse_category(category_str):
    """Extract category ID from gold strings like 'Category 3: Defining problem'.

    Notes:
    - The shared task label space is Category 1-9.
    - Gold annotations may include Category 10/11 (e.g., legal/insurance, other);
      these are treated as out-of-scope and mapped to 0 so they are ignored.
    """
    match = re.match(r'Category\s*(\d+)', category_str)
    if match:
        cat = int(match.group(1))
        # print(f"VALID Category: {cat}")
        return cat if cat in VALID_CATEGORY_IDS else 0
    return 0

def get_raw_text(data_dir, filename):
    # Filename example: 19326_157593_4002.json
    # We want to find 19326_157593_4002*.txt in raw_text dir
    basename = os.path.splitext(filename)[0]
    
    raw_text_path = os.path.join(data_dir, 'raw_text', f'{basename}.txt')
    return open(raw_text_path).read()

def refine_span(text, start, end):
    """
    Refines span boundaries:
    1. Expands to full word boundaries if splitting a word.
    2. Shrinks by removing leading/trailing punctuation.
    """
    # 1. Expand to word boundaries
    # Expand left: if we are inside a word (char before and char at start are alnum)
    while start > 0 and start < len(text) and text[start-1].isalnum() and text[start].isalnum():
        start -= 1
        
    # Expand right: if we are inside a word
    while end > 0 and end < len(text) and text[end-1].isalnum() and text[end].isalnum():
        end += 1

    # 2. Shrink (remove punctuation)
    span_text = text[start:end]
    if not span_text.strip():
        return None, None
        
    spans = list(tokenizer.span_tokenize(span_text))
    
    if not spans:
        return None, None

    words = [span_text[s:e] for s, e in spans]
    
    # Helper to check if a word is content (alphanumeric or part of MIMIC tag)
    def is_content(w):
        return any(c.isalnum() for c in w) or '**' in w

    # Find start
    idx_start = 0
    while idx_start < len(words):
        word = words[idx_start]
            
        if is_content(word):
            break
            
        # Handle special punctuation start '[' if followed by '**'
        if word == '[' and idx_start + 1 < len(words) and '**' in words[idx_start + 1]:
            break
            
        # Otherwise, it's punctuation to remove
        idx_start += 1

    # Find end
    idx_end = len(words)
    while idx_end > idx_start:
        word = words[idx_end-1]
            
        if is_content(word):
            break
            
        # Handle special punctuation end ']' if preceded by '**'
        if word == ']' and idx_end - 2 >= 0 and '**' in words[idx_end - 2]:
            break
            
        # Otherwise, it's punctuation to remove
        idx_end -= 1
            
    if idx_start >= idx_end:
        return None, None
        
    new_start = start + spans[idx_start][0]
    new_end = start + spans[idx_end-1][1]
    
    return new_start, new_end

def sort_annotations_by_offset(annotations: list[dict]) -> list[dict]:
    """
    Sort annotations by start_offset then end_offset (ascending), robust to missing fields.
    """
    def key(a):
        try:
            s = int(a.get("start_offset", 0))
        except Exception:
            s = 0
        try:
            e = int(a.get("end_offset", 0))
        except Exception:
            e = 0
        return (s, e)
    return sorted(annotations, key=key)

# ------------------------------------
# Handling overlapping annotations
# ------------------------------------

def _intervals_overlap(a, b) -> bool:
    return max(a["start"], b["start"]) < min(a["end"], b["end"])

def _is_nested(outer, inner) -> bool:
    # "completely nested": proper containment (not equal)
    return (
        outer["start"] <= inner["start"]
        and inner["end"] <= outer["end"]
        and (outer["start"] != inner["start"] or outer["end"] != inner["end"])
    )

def check_overlaps(spans: list[dict]) -> list[tuple[int, int, str]]:
    """
    Return list of overlapping pairs as (i, j, overlap_type) where i < j.

    overlap_type categories:
      - same_class_nested
      - multi_class_nested
      - same_class_overlap (but not nested)
      - remaining_other
    """
    pairs = []
    spans = sorted(spans, key=lambda x: (x["start"], x["end"]))
    active = []

    for j, cur in enumerate(spans):
        active = [i for i in active if spans[i]["end"] > cur["start"]]
        for i in active:
            a = spans[i]
            b = cur
            if not _intervals_overlap(a, b):
                continue

            if a["cat"] == b["cat"]:
                if _is_nested(a, b) or _is_nested(b, a):
                    t = "same_class_nested"
                else:
                    t = "same_class_overlap"
            else:
                if _is_nested(a, b) or _is_nested(b, a):
                    t = "multi_class_nested"
                else:
                    t = "remaining_other"
            pairs.append((i, j, t))
        active.append(j)

    return pairs

def _dedup_exact(spans: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for s in spans:
        k = (s["start"], s["end"], s["cat"])
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out

def remove_same_cat_nested(spans: list[dict]) -> list[dict]:
    """
    For nested spans with SAME category: remove the shorter (keep the longer).
    """
    spans = sorted(spans, key=lambda x: (x["cat"], x["start"], -(x["end"] - x["start"]), x["end"]))
    out = []
    # Process per category
    by_cat = defaultdict(list)
    for s in spans:
        by_cat[s["cat"]].append(s)

    for cat, lst in by_cat.items():
        lst = sorted(lst, key=lambda x: (x["start"], -(x["end"]), x["end"]))
        kept = []
        for s in lst:
            # if s is contained in any kept span, drop it
            drop = False
            for k in kept:
                if k["start"] <= s["start"] and s["end"] <= k["end"]:
                    drop = True
                    break
            if not drop:
                kept.append(s)
        out.extend(kept)

    return sorted(out, key=lambda x: (x["start"], x["end"], x["cat"]))

def _union_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]

def split_outer_by_nested_diff_cats(spans: list[dict]) -> list[dict]:
    """
    For completely nested spans with DIFFERENT categories:
      - keep the shorter (inner) span as-is
      - split the longer (outer) span into parts excluding all nested (diff-cat) inners
    Works with multiple nested spans (diff cats) inside a long outer span.
    """
    spans = sorted(spans, key=lambda x: (x["start"], x["end"], x["cat"]))
    n = len(spans)

    # Precompute for each outer span, the list of inner intervals (diff cats) fully contained
    contained = [[] for _ in range(n)]
    for i in range(n):
        outer = spans[i]
        for j in range(n):
            if i == j:
                continue
            inner = spans[j]
            if outer["cat"] != inner["cat"] and _is_nested(outer, inner):
                contained[i].append((inner["start"], inner["end"]))

    new_spans = []
    for i, outer in enumerate(spans):
        inners = _union_intervals(contained[i])

        # If no contained diff-cat spans, keep as-is
        if not inners:
            new_spans.append(outer)
            continue

        # Subtract union of inner intervals from [outer.start, outer.end)
        cursor = outer["start"]
        parts = []
        for s, e in inners:
            if s > cursor:
                parts.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < outer["end"]:
            parts.append((cursor, outer["end"]))

        # Emit split parts (skip empty)
        part_idx = 0
        for ps, pe in parts:
            if pe <= ps:
                continue
            part = deepcopy(outer)
            part["start"] = ps
            part["end"] = pe
            # Keep ann in sync so it's never stale
            part["ann"]["start_offset"] = ps
            part["ann"]["end_offset"] = pe
            if part.get("annotation_id") is not None:
                part["annotation_id"] = f"{part['annotation_id']}_part{part_idx}"
            part_idx += 1
            new_spans.append(part)  

    return sorted(_dedup_exact(new_spans), key=lambda x: (x["start"], x["end"], x["cat"]))

def merge_same_cat_overlaps(spans: list[dict]) -> list[dict]:
    """
    For overlapping spans with SAME category: merge them into one.
    (Also merges “touching” spans where next.start <= cur.end.)
    """
    by_cat = defaultdict(list)
    for s in spans:
        by_cat[s["cat"]].append(s)

    merged_all = []
    for cat, lst in by_cat.items():
        lst = sorted(lst, key=lambda x: (x["start"], x["end"]))
        cur = None
        for s in lst:
            if cur is None:
                cur = deepcopy(s)
                continue
            if s["start"] <= cur["end"]:  # overlap or touch
                cur["end"] = max(cur["end"], s["end"])
                # keep a stable annotation_id (optional)
            else:
                merged_all.append(cur)
                cur = deepcopy(s)
        if cur is not None:
            merged_all.append(cur)

    return sorted(_dedup_exact(merged_all), key=lambda x: (x["start"], x["end"], x["cat"]))

def annotations_to_spans(annotations: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split annotations into:
      - in_scope spans list (cat 1–9) with normalized fields
      - out_of_scope annotations (kept but not used for overlap logic)
    """
    in_scope = []
    out_scope = []
    for a in annotations:
        try:
            s = int(a.get("start_offset", 0))
            e = int(a.get("end_offset", 0))
        except Exception:
            out_scope.append(a)
            continue
        if e < s:
            s, e = e, s

        cat = parse_category(a.get("category"))
        if cat in VALID_CATEGORY_IDS:
            in_scope.append(
                {
                    "start": s,
                    "end": e,
                    "cat": cat,
                    "ann": a,  # pointer to original dict (we'll deepcopy when writing)
                }
            )
        else:
            out_scope.append(a)
    return in_scope, out_scope

def spans_to_annotations(spans: list[dict]) -> list[dict]:
    """
    Convert internal spans back to annotation dicts, preserving other keys.
    """
    out = []
    for s in spans:
        a = deepcopy(s["ann"])
        a["start_offset"] = int(s["start"])
        a["end_offset"] = int(s["end"])
        out.append(a)
    return out

def normalize_and_fix_overlaps(annotations: list[dict]) -> tuple[list[dict], dict]:
    """
    Implements steps:
      1) sort by offsets
      2) nested handling rules
      3) same-cat overlap merge
      4) recheck overlaps

    Returns:
      - new_annotations (in-scope processed + out-of-scope untouched)
      - overlap_report dict with counts:
          same_class_nested, multi_class_nested, same_class_overlap, remaining_other, remaining_after
    """
    annotations_sorted = sort_annotations_by_offset(annotations)
    spans, out_scope = annotations_to_spans(annotations_sorted)

    # Initial overlap classification (counts are PAIR COUNTS)
    init_pairs = check_overlaps(spans)
    init_counts = defaultdict(int)
    for _, _, t in init_pairs:
        init_counts[t] += 1

    # Step 2a: same-cat nested -> drop shorter
    spans = remove_same_cat_nested(spans)

    # Step 2b: diff-cat nested -> split outer(s) around inner(s)
    spans = split_outer_by_nested_diff_cats(spans)

    # Step 3: same-cat overlaps -> merge
    spans = merge_same_cat_overlaps(spans)

    # Step 4: recheck overlaps after cleanup
    final_pairs = check_overlaps(spans)
    remaining_after = len(final_pairs)

    # Rebuild annotations: in-scope (processed) + out-of-scope (unchanged)
    new_annotations = spans_to_annotations(spans) + out_scope
    new_annotations = sort_annotations_by_offset(new_annotations)  # final tidy sort

    report = {
        "same_class_nested": int(init_counts["same_class_nested"]),
        "multi_class_nested": int(init_counts["multi_class_nested"]),
        "same_class_overlap": int(init_counts["same_class_overlap"]),
        "remaining_other": int(init_counts["remaining_other"]),
        "remaining_after_cleanup": int(remaining_after),
    }
    return new_annotations, report

# ------------------------------------
# End of  overlapping annotations
# ------------------------------------

def process_json_file(json_path, data_dir,overlap_rows: list[dict]):
    filename = os.path.basename(json_path)
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {json_path}")
        return None

    # Handle if data is list or dict
    if isinstance(data, list):
        if not data:
            return None
        content = data[0]
        is_list_wrapper = True
    else:
        content = data
        is_list_wrapper = False

    if 'annotations' not in content:
        print(f"No annotations found in {json_path}")
        return None

    text = get_raw_text(data_dir, filename)
    if text is None:
        print(f"Raw text not found for {filename}")
        return None
    fixed_annotations, overlap_report = normalize_and_fix_overlaps(content["annotations"])

    file_id = os.path.splitext(filename)[0]
    overlap_rows.append({"file_id": file_id, **overlap_report})
    new_annotations = []

    for annot in fixed_annotations:
        try:
            start = int(annot['start_offset'])
            end = int(annot['end_offset'])
        except (ValueError, KeyError):
            continue
            
        new_start, new_end = refine_span(text, start, end)
        
        if new_start is not None:
            annot['start_offset'] = new_start
            annot['end_offset'] = new_end
            new_annotations.append(annot)
            
    content['annotations'] = new_annotations
    
    if is_list_wrapper:
        return [content]
    return content

def main():
    parser = argparse.ArgumentParser(description='Clean MIMIC decisions data')
    parser.add_argument('--data_dir', type=str, default='MedDec', 
                      help='Directory containing data and raw_text folders')
    parser.add_argument(
        "--overlap_report_csv",
        type=str,
        default="overlap_report.csv",
        help="Where to save the per-file overlap report (CSV). Saved under data_dir by default if relative.",
    )
    args = parser.parse_args()
    
    data_path = os.path.join(args.data_dir, 'data_non_overlapped')
    raw_text_path = os.path.join(args.data_dir, 'raw_text')
    unclean_data_path = os.path.join(args.data_dir, 'data_unclean')
    
    if not os.path.exists(data_path):
        print(f"Data directory not found: {data_path}")
        return
        
    if not os.path.exists(raw_text_path):
        print(f"Raw text directory not found: {raw_text_path}")
        return
        
    # Move existing data to data_unclean
    if os.path.exists(unclean_data_path):
        print(f"Warning: {unclean_data_path} already exists. Merging/Overwriting?")
        pass
    else:
        print(f"Moving {data_path} to {unclean_data_path}...")
        shutil.move(data_path, unclean_data_path)
    
    # Create new data directory
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    # Iterate over files in data_unclean
    json_files = glob.glob(os.path.join(unclean_data_path, '*.json'))
    print(f"Found {len(json_files)} files to process.")
    
    overlap_rows = []
    count = 0
    for json_file in tqdm(json_files):
        cleaned_content = process_json_file(json_file, args.data_dir, overlap_rows)
        
        if cleaned_content is not None:
            basename = os.path.basename(json_file)
            output_file = os.path.join(data_path, basename)
            
            with open(output_file, 'w') as f:
                json.dump(cleaned_content, f, indent=4)
            count += 1
    df = pd.DataFrame(overlap_rows, columns=[
        "file_id",
        "same_class_nested",
        "multi_class_nested",
        "same_class_overlap",
        "remaining_other",
        "remaining_after_cleanup",
    ]).fillna(0)

    # If relative path, save under data_dir
    report_path = args.overlap_report_csv
    if not os.path.isabs(report_path):
        report_path = os.path.join(args.data_dir, report_path)
    df.to_csv(report_path, index=False)       
    print(f"Processed and saved {count} files to {data_path}")
    print(f"Saved overlap report: {report_path}")

if __name__ == "__main__":
    main()
