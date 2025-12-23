import os
import glob
import json
import argparse
import shutil
import numpy as np
import nltk
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


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
        
    tokenizer = TreebankWordTokenizer()
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

def process_json_file(json_path, data_dir):
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

    new_annotations = []
    
    for annot in content['annotations']:
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
    
    args = parser.parse_args()
    
    data_path = os.path.join(args.data_dir, 'data')
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
    
    count = 0
    for json_file in tqdm(json_files):
        cleaned_content = process_json_file(json_file, args.data_dir)
        
        if cleaned_content is not None:
            basename = os.path.basename(json_file)
            output_file = os.path.join(data_path, basename)
            
            with open(output_file, 'w') as f:
                json.dump(cleaned_content, f, indent=4)
            count += 1
            
    print(f"Processed and saved {count} files to {data_path}")

if __name__ == "__main__":
    main()
