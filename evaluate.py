"""
MedDec Shared Task Evaluation Script

Evaluates predictions against gold labels using:
1. Span F1: Compares predicted spans to gold spans with preprocessing
2. Token F1: Token-level classification F1 with macro-averaging

Usage:
    python evaluate.py --gold_dir ./data/ --predictions predictions.json --raw_text_dir ./raw_text/
"""

import json
import argparse
import csv
import os
import re
import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer

VALID_CATEGORY_IDS = set(range(1, 10))  # Shared task label space: 1–9 only.

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, OSError):
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass


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
        return cat if cat in VALID_CATEGORY_IDS else 0
    return 0


def parse_ids_from_filename(file_name):
    """Parse SUBJECT_ID, HADM_ID, ROW_ID from a MedDec discharge summary id.
    
    Expected formats (examples):
    - 78_100536_1787
    - 78_100536_1787-annotator
    - /path/to/78_100536_1787.json
    """
    base = os.path.basename(str(file_name))
    base = os.path.splitext(base)[0]
    parts = base.split('_')
    if len(parts) < 3:
        return None
    try:
        sid = int(parts[0])
        hadm = int(parts[1])
        rid_part = parts[2]
        rid = int(rid_part.split('-')[0])
    except (TypeError, ValueError):
        return None
    return sid, hadm, rid


def _normalize_sex(gender):
    g = str(gender or '').strip().upper()
    if g in {'F', 'FEMALE'}:
        return 'Female'
    if g in {'M', 'MALE'}:
        return 'Male'
    return 'Other'


def _normalize_language(language):
    l = str(language or '').strip().upper()
    if l == 'ENGL' or l == 'ENGLISH':
        return 'English'
    return 'Non-English'


def _normalize_race(ethnicity):
    """Map MedDec/MIMIC ethnicity buckets to shared-task race groups."""
    e = str(ethnicity or '').strip().upper()

    # MedDec generate_stats.py mapping outputs: W, AA, Hisp., Asian, NH, OTHER,
    # UNKNOWN/NOT SPECIFIED, UNABLE TO OBTAIN, PATIENT DECLINED TO ANSWER, ...
    if e == 'W' or 'WHITE' in e:
        return 'White'
    if e == 'AA' or 'BLACK' in e or 'AFRICAN' in e:
        return 'African American'
    if e.startswith('HISP'):
        return 'Hispanic'
    if e == 'ASIAN' or 'ASIAN' in e:
        return 'Asian'
    return 'Other'


def load_stats_csv(stats_csv_path):
    """Load stats.csv as a mapping from (SUBJECT_ID, HADM_ID, ROW_ID) -> group labels."""
    stats = {}
    with open(stats_csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sid = int(row.get('SUBJECT_ID', ''))
                hadm = int(row.get('HADM_ID', ''))
                rid = int(row.get('ROW_ID', ''))
            except (TypeError, ValueError):
                continue

            ethnicity = row.get('ETHNICITY', '')
            language = row.get('LANGUAGE', '')
            gender = row.get('GENDER', '')

            stats[(sid, hadm, rid)] = {
                'sex': _normalize_sex(gender),
                'race': _normalize_race(ethnicity),
                'language': _normalize_language(language),
                'raw': {
                    'ETHNICITY': ethnicity,
                    'LANGUAGE': language,
                    'GENDER': gender,
                },
            }
    return stats


def build_refined_span_sets(gold_annotations, predictions, raw_texts):
    """Precompute refined span sets per file for faster subgroup evaluation."""
    gold_sets = {}
    pred_sets = {}

    for file_name, annotations in gold_annotations.items():
        raw_text = raw_texts.get(file_name, "")
        spans = set()
        for ann in annotations:
            start = int(ann.get('start_offset', 0))
            end = int(ann.get('end_offset', 0))
            label = parse_category(ann.get('category', 0))
            if label == 0:
                continue

            refined_start, refined_end = refine_span(raw_text, start, end)
            if refined_start is not None:
                refined_text = raw_text[refined_start:refined_end]
                spans.add((label, refined_text.lower()))
        gold_sets[file_name] = spans

    for file_name, preds in predictions.items():
        raw_text = raw_texts.get(file_name, "")
        spans = set()
        for pred in preds:
            start = int(pred.get('start_offset', 0))
            end = int(pred.get('end_offset', 0))
            label = int(pred.get('category', 0))
            if label not in VALID_CATEGORY_IDS:
                continue

            refined_start, refined_end = refine_span(raw_text, start, end)
            if refined_start is not None:
                refined_text = raw_text[refined_start:refined_end]
                spans.add((label, refined_text.lower()))
        pred_sets[file_name] = spans

    return gold_sets, pred_sets


def compute_span_f1_from_sets(gold_sets_by_file, pred_sets_by_file, files):
    """Compute span F1 from precomputed per-file span sets."""
    tp = fp = fn = 0
    for file_name in files:
        gold_spans = gold_sets_by_file.get(file_name, set())
        pred_spans = pred_sets_by_file.get(file_name, set())
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def compute_token_f1_by_file(gold_annotations, predictions, raw_texts):
    """Compute token-level F1 per file (macro over classes within a file)."""
    per_file = {}

    all_files = set(gold_annotations.keys()) | set(predictions.keys())
    for file_name in all_files:
        raw_text = raw_texts.get(file_name)
        if not raw_text:
            continue

        tokens = tokenize_text(raw_text)
        if not tokens:
            continue

        gold_anns = gold_annotations.get(file_name, [])
        pred_anns = predictions.get(file_name, [])

        gold_labels = assign_token_labels(tokens, gold_anns, is_gold=True)
        pred_labels = assign_token_labels(tokens, pred_anns, is_gold=False)

        per_file[file_name] = compute_token_f1_per_sample(gold_labels, pred_labels)

    return per_file


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def compute_shared_task_scores(gold_annotations, predictions, raw_texts, stats_csv_path, overall_span_f1, overall_token_f1):
    """Compute shared-task final score with subgroup worst-case adjustment."""
    stats = load_stats_csv(stats_csv_path)

    # Build mapping from file -> group labels (based on gold set)
    file_groups = {}
    missing = []
    for file_name in gold_annotations.keys():
        ids = parse_ids_from_filename(file_name)
        if ids is None or ids not in stats:
            missing.append(file_name)
            continue
        file_groups[file_name] = {
            'sex': stats[ids]['sex'],
            'race': stats[ids]['race'],
            'language': stats[ids]['language'],
        }

    # Precompute sets/scores once, then aggregate by subgroup.
    gold_span_sets, pred_span_sets = build_refined_span_sets(gold_annotations, predictions, raw_texts)
    token_f1_per_file = compute_token_f1_by_file(gold_annotations, predictions, raw_texts)

    def files_for(dim, group_value):
        return {fn for fn, g in file_groups.items() if g.get(dim) == group_value}

    subgroup_defs = {
        'sex': ['Female', 'Male'],
        'race': ['White', 'African American', 'Hispanic', 'Asian', 'Other'],
        'language': ['English', 'Non-English'],
    }

    subgroup_scores = {'sex': {}, 'race': {}, 'language': {}}
    base_scores = []

    for dim, groups in subgroup_defs.items():
        for group_value in groups:
            files = files_for(dim, group_value)
            if not files:
                continue

            span_m = compute_span_f1_from_sets(gold_span_sets, pred_span_sets, files)
            tok = mean(token_f1_per_file[fn] for fn in files if fn in token_f1_per_file)
            base = (span_m['f1'] + tok) / 2

            n_spans = sum(len(gold_span_sets.get(fn, set())) for fn in files)


            subgroup_scores[dim][group_value] = {
                'n_files': len(files),
                'n_spans': n_spans,
                'span_f1': span_m['f1'],
                'token_f1': tok,
                'base_score': base,
            }
            base_scores.append(base)

    base_score = (overall_span_f1 + overall_token_f1) / 2
    worst_group_score = min(base_scores) if base_scores else 0.0
    final_score = (base_score + worst_group_score) / 2

    return {
        'base_score': base_score,
        'worst_group_score': worst_group_score,
        'final_score': final_score,
        'subgroups': subgroup_scores,
    }


def load_gold_annotations(gold_file):
    """Load gold annotations from a JSON file."""
    with open(gold_file, 'r') as f:
        data = json.load(f)
    
    # Handle both single file and aggregated formats
    if isinstance(data, dict) and 'annotations' in data:
        # Single file format
        file_name = data.get('discharge_summary_id', os.path.basename(gold_file).replace('.json', ''))
        file_name = file_name.rstrip("_")
        return {file_name: data['annotations']}
    elif isinstance(data, list):
        # List of files
        result = {}
        for item in data:
            if isinstance(item, dict) and 'discharge_summary_id' in item:
                file_name = item['discharge_summary_id']
                file_name = file_name.rstrip("_")
                result[file_name] = item.get('annotations', [])
        return result
    return {}


def load_predictions(predictions_file):
    """Load predictions from a JSON file.
    
    Expected format:
    [
        {
            "file_name": "78_100536_1787",
            "predictions": [
                {"start_offset": 322, "end_offset": 332, "category": 3}
            ]
        }
    ]
    """
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    result = {}
    for item in data:
        file_name = item['file_name']
        result[file_name] = item.get('predictions', [])
    return result


def load_raw_text(raw_text_dir, file_name):
    """Load raw text for a given file name."""
    path = os.path.join(raw_text_dir, f"{file_name}.txt")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Raw text file not found: {path}")


def refine_span(text, start, end):
    """
    Refines span boundaries:
    1. Expands to full word boundaries if splitting a word.
    2. Shrinks by removing leading/trailing punctuation.
    """
    if not text:
        return None, None
        
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
        
    tokenizer = TreebankWordTokenizer()
    spans = list(tokenizer.span_tokenize(span_text))
        
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


def compute_span_f1(gold_annotations, predictions, raw_texts):
    """
    Compute Span F1 score.
    
    A prediction matches a gold span if:
    - Same file
    - Same label
    - Preprocessed spans match
    """
    gold_spans = set()
    pred_spans = set()
    
    for file_name, annotations in gold_annotations.items():
        raw_text = raw_texts.get(file_name, "")
        for ann in annotations:
            start = int(ann.get('start_offset', 0))
            end = int(ann.get('end_offset', 0))
            label = parse_category(ann.get('category', 0))
            if label == 0:
                continue
            
            refined_start, refined_end = refine_span(raw_text, start, end)
            if refined_start is not None:
                refined_text = raw_text[refined_start:refined_end]
                gold_spans.add((file_name, label, refined_text.lower()))
    
    for file_name, preds in predictions.items():
        raw_text = raw_texts.get(file_name, "")
        for pred in preds:
            start = int(pred.get('start_offset', 0))
            end = int(pred.get('end_offset', 0))
            label = int(pred.get('category', 0))
            if label not in VALID_CATEGORY_IDS:
                continue
            
            refined_start, refined_end = refine_span(raw_text, start, end)
            if refined_start is not None:
                refined_text = raw_text[refined_start:refined_end]
                pred_spans.add((file_name, label, refined_text.lower()))
    
    # Compute metrics
    tp = len(gold_spans & pred_spans)
    fp = len(pred_spans - gold_spans)
    fn = len(gold_spans - pred_spans)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def tokenize_text(text):
    """Word tokenization using NLTK with character offsets."""
    tokens = []
    nltk_tokens = word_tokenize(text)
    
    # Reconstruct character offsets by finding each token in the text
    current_pos = 0
    for token in nltk_tokens:
        # Find the token in the remaining text
        idx = text.find(token, current_pos)
        if idx != -1:
            tokens.append({
                'text': token,
                'start': idx,
                'end': idx + len(token)
            })
            current_pos = idx + len(token)
    
    return tokens


def assign_token_labels(tokens, annotations, is_gold=True):
    """Assign labels to tokens based on spans.
    
    If a token overlaps with multiple spans, use the first one encountered.
    """
    labels = [0] * len(tokens)
    
    for ann in annotations:
        start = int(ann.get('start_offset', 0))
        end = int(ann.get('end_offset', 0))
        if is_gold:
            label = parse_category(ann.get('category', 0))
        else:
            label = int(ann.get('category', 0))

        if label not in VALID_CATEGORY_IDS:
            continue
        
        for i, token in enumerate(tokens):
            # Check if token overlaps with annotation span
            if token['start'] < end and token['end'] > start:
                if labels[i] == 0:  # Only assign if not already labeled
                    labels[i] = label
    
    return labels


def compute_token_f1_per_sample(gold_labels, pred_labels):
    """Compute token-level F1 for a single sample (macro-averaged over classes)."""
    # Get all unique labels (excluding O)
    all_labels = set(gold_labels + pred_labels)
    all_labels.discard(0)
    
    if not all_labels:
        return 1.0 if gold_labels == pred_labels else 0.0
    
    f1_scores = []
    for label in all_labels:
        tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == label and p != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def compute_token_f1(gold_annotations, predictions, raw_texts):
    """
    Compute Token F1 score with macro-averaging across samples.
    """
    sample_f1_scores = []
    
    # Get all file names from both gold and predictions
    all_files = set(gold_annotations.keys()) | set(predictions.keys())
    
    for file_name in all_files:
        raw_text = raw_texts.get(file_name)
        if not raw_text:
            continue
        
        tokens = tokenize_text(raw_text)
        if not tokens:
            continue
        
        gold_anns = gold_annotations.get(file_name, [])
        pred_anns = predictions.get(file_name, [])
        
        gold_labels = assign_token_labels(tokens, gold_anns, is_gold=True)
        pred_labels = assign_token_labels(tokens, pred_anns, is_gold=False)
        
        f1 = compute_token_f1_per_sample(gold_labels, pred_labels)
        sample_f1_scores.append(f1)
    
    return sum(sample_f1_scores) / len(sample_f1_scores) if sample_f1_scores else 0.0


def main():
    parser = argparse.ArgumentParser(description='MedDec Shared Task Evaluation')
    parser.add_argument('--gold_dir', required=True, help='Directory containing gold annotation files')
    parser.add_argument('--predictions', required=True, help='Path to predictions JSON file')
    parser.add_argument('--raw_text_dir', required=True, help='Directory containing raw text files')
    parser.add_argument('--gold_file', help='Optional: specific gold file (overrides gold_dir)')
    parser.add_argument('--stats_csv',
                        help='Optional: path to stats.csv for shared-task final scoring')
    parser.add_argument('--split_file', help='Optional: path to split file (e.g. splits/val.txt)')
    parser.add_argument('--output', help='Optional: output file for results (JSON)')
    args = parser.parse_args()
    
    # Load gold annotations
    if args.gold_file:
        gold_annotations = load_gold_annotations(args.gold_file)
    else:
        gold_annotations = {}
        for fn in os.listdir(args.gold_dir):
            if fn.endswith('.json'):
                file_anns = load_gold_annotations(os.path.join(args.gold_dir, fn))
                gold_annotations.update(file_anns)
    
    # Load predictions
    predictions = load_predictions(args.predictions)
    
    # Filter by split file if provided
    if args.split_file:
        split_ids = set()
        with open(args.split_file, 'r') as f:
            for line in f:
                line = line.strip()
                split_ids.add(line)
        
        gold_annotations = {k: v for k, v in gold_annotations.items() if k in split_ids}
        predictions = {k: v for k, v in predictions.items() if k in split_ids}
        print(f"Filtered to {len(gold_annotations)} gold files and {len(predictions)} prediction files based on {args.split_file}")

    # Load raw texts
    raw_texts = {}
    all_files = set(gold_annotations.keys()) | set(predictions.keys())
    for file_name in all_files:
        text = load_raw_text(args.raw_text_dir, file_name)
        raw_texts[file_name] = text
    
    # Compute metrics
    span_metrics = compute_span_f1(gold_annotations, predictions, raw_texts)
    token_f1 = compute_token_f1(gold_annotations, predictions, raw_texts)

    shared_task = None
    if args.stats_csv:
        shared_task = compute_shared_task_scores(
            gold_annotations=gold_annotations,
            predictions=predictions,
            raw_texts=raw_texts,
            stats_csv_path=args.stats_csv,
            overall_span_f1=span_metrics['f1'],
            overall_token_f1=token_f1,
        )
    
    # Output results
    results = {
        'span_f1': {
            'precision': span_metrics['precision'],
            'recall': span_metrics['recall'],
            'f1': span_metrics['f1']
        },
        'token_f1': token_f1,
        'statistics': {
            'gold_files': len(gold_annotations),
            'prediction_files': len(predictions),
            'files_with_text': len(raw_texts)
        }
    }

    if shared_task is not None:
        results['shared_task'] = shared_task
    
    print("=" * 50)
    print("MedDec Shared Task Evaluation Results")
    print("=" * 50)
    print(f"\nSpan F1:\n  {span_metrics['f1']:.4f}")
    print(f"\nToken F1:\n  {token_f1:.4f}")

    if shared_task is not None:
        print("\nShared-task final scoring (uses stats.csv):")
        print(f"  Base score:        {shared_task['base_score']:.4f}")
        print(f"  Worst group score: {shared_task['worst_group_score']:.4f}")
        print(f"  Final score:       {shared_task['final_score']:.4f}")

        # Print a compact breakdown to help debugging.
        print("\nWorst-group candidates (base_score by subgroup):")
        for dim in ['sex', 'race', 'language']:
            groups = shared_task.get('subgroups', {}).get(dim, {})
            if not groups:
                continue
            for group_value, m in groups.items():
                print(f"  - {dim:8s} = {group_value:20s} n={m['n_spans']:3d}  base={m['base_score']:.4f}  span_f1={m['span_f1']:.4f}  token_f1={m['token_f1']:.4f}")

    print("=" * 50)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == '__main__':
    main()
