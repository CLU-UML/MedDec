import os
import json
import torch
import numpy as np
from os import path
from glob import glob
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from collections import defaultdict
set_seed(0)


categories = [
        "Contact related: Decision regarding admittance or discharge from hospital, scheduling of control and referral to other parts of the healthcare system",
        "Gathering additional information: Decision to obtain information from other sources than patient interview, physical examination and patient chart",
        "Defining problem: Complex, interpretative assessments that define what the problem is and reflect a medically informed conclusion",
        "Treatment goal: Decision to set defined goal for treatment and thereby being more specific than giving advice",
        "Drug: Decision to start, refrain from, stop, alter or maintain a drug regimen",
        "Therapeutic procedure related: Decision to intervene on a medical problem, plan, perform or refrain from therapeutic procedures of a medical nature",
        "Evaluating test result: Simple, normative assessments of clinical findings and tests",
        "Deferment: Decision to actively delay decision or a rejection to decide on a problem presented by a patient",
        "Advice and precaution: Decision to give the patient advice or precaution, thereby transferring responsibility for action from the provider to the patient",
        "Legal and insurance related: Medical decision concerning the patient, which is based on or restricted by legal regulations or financial arrangements",
        ]


data_dir = '/data/mohamed/data/mimic_decisions/'
test_samples = [x.strip() for x in open(path.join(data_dir, 'test.txt'))]
np.random.shuffle(test_samples)

def resolve_src(fn):
    """
    Resolves the source text file corresponding to the given filename.
    This function takes a filename, extracts its basename, and searches for a 
    corresponding text file in the 'raw_text' directory within the specified 
    data directory. It reads and returns the content of the first matching 
    text file.

    Args:
        fn (str): The filename to resolve the source text for.

    Returns:
        str: The content of the resolved text file.
    """
    basename = path.basename(fn).split("-")[0]
    txt_candidates = glob(os.path.join(data_dir,
        f'raw_text/{basename}*.txt'))
    text = open(txt_candidates[0]).read()
    return text

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        )

def prompt(note, cat):
    """
    Extracts sub-strings from a clinical note that contain medical decisions within a specified category.

    Args:
        note (str): The clinical note from which to extract sub-strings.
        cat (int): The category index for which to extract medical decisions.

    Returns:
        str: Extracted sub-strings, each on a new line. If no such sub-strings exist, returns "None".
    """
    messages = [
            {'role': 'system', 'content': f'Extract all sub-strings from the following Clinical Note that contain medical decisions within the specified category.\nPrint each sub-string in a new line.\nIf no such sub-string exists, output \"None\".\n[Clinical Note]: {note}'},
            {"role": "user", "content": f"[Category]: {categories[cat-1]}"},
            # {"role": "user", "content": f"Extract all sub-strings from the following Clinical Note that contain medical decisions within the specified category.\nPrint each sub-string in a new line.\nIf no such sub-string exists, output \"None\".\n[Clinical Note]: {note}\n\n[Category]: {categories[cat-1]}"},
            ]

    input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
            ).to(model.device)

    terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

    outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def prompt_oneshot(note, cat, demo_cat, demos):
    """
    Generates a response from a clinical note based on specified categories and demonstrations.

    Args:
        note (str): The clinical note from which to extract sub-strings.
        cat (int): The category index for the current extraction.
        demo_cat (int): The category index for the demonstration extraction.
        demos (str): The demonstration examples to guide the extraction.

    Returns:
        str: The generated response containing sub-strings of medical decisions within the specified category.
    """
    messages = [
            {'role': 'system', 'content': f'Extract all sub-strings from the following Clinical Note that contain medical decisions within the specified category.\nPrint each sub-string in a new line.\nIf no such sub-string exists, output \"None\".\n[Clinical Note]: {note}'},
            {"role": "user", "content": f"[Category]: {categories[demo_cat-1]}"},
            {"role": "assistant", "content": f"{demos}"},
            {"role": "user", "content": f"[Category]: {categories[cat-1]}"},
            ]

    input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
            ).to(model.device)

    terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

    outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def zeroshot():
    """
    Processes a subset of test samples using a zero-shot approach.

    For each of the first 10 test samples, this function:
    1. Loads annotations from a JSON file.
    2. Groups the annotations.
    3. Creates an output directory for the results.
    4. Saves the grouped annotations to a JSON file in the output directory.
    5. For each category from 1 to 9:
        a. Resolves the source text.
        b. Generates a response using a prompt.
        c. Writes the response to a file in the output directory.

    If a CUDA OutOfMemoryError occurs, it prints 'OOM' and continues with the next sample.

    Raises:
        torch.cuda.OutOfMemoryError: If the GPU runs out of memory during processing.

    Note:
        This function assumes the existence of several helper functions and variables:
        - `test_samples`: A list of test sample filenames.
        - `data_dir`: The directory containing the data files.
        - `group_annots`: A function to group annotations.
        - `resolve_src`: A function to resolve the source text from a filename.
        - `prompt`: A function to generate a response given a text and category.
    """
    for i, fn in enumerate(test_samples[:10]):
        print(i)
        try:
            annots = json.load(open(path.join(data_dir, 'data', fn)))[0]['annotations']
            annots = group_annots(annots)
            out_dir = path.join('gens', 'zero', str(i))
            os.makedirs(out_dir, exist_ok=True)
            json.dump(annots, open(path.join(out_dir, 'labels.json'), 'w'))
            for cat in range(1, 10):
                text = resolve_src(fn)
                response = prompt(text, cat)
                with open(path.join(out_dir, f'cat_{cat}'), 'w') as f:
                    f.write(response)
        except torch.cuda.OutOfMemoryError:
            print('OOM')
            continue

def parse_cat(cat):
    """
    Parses a string to extract the first numeric value.

    Args:
        cat (str): The input string containing numeric and non-numeric characters.

    Returns:
        int or None: The first numeric value found in the string as an integer. 
                     If no numeric value is found, returns None.
    """
    for i,c in enumerate(cat):
        if c.isnumeric():
            if cat[i+1].isnumeric():
                return int(cat[i:i+2])
            return int(c)
    return None

def group_annots(annots):
    """
    Groups annotations by their parsed category.

    Args:
        annots (list of dict): A list of annotation dictionaries. Each dictionary
                               should have at least the keys 'category' and 'decision'.

    Returns:
        defaultdict: A defaultdict where the keys are parsed categories and the values
                     are lists of decisions corresponding to those categories.

    Notes:
        - The function uses `parse_cat` to parse the category of each annotation.
        - Annotations with categories that cannot be parsed (i.e., `parse_cat` returns None)
          are skipped.
    """
    new_annots = defaultdict(list)
    for ann in annots:
        cat = parse_cat(ann['category'])
        if cat is None:
            continue
        dec = ann['decision']
        new_annots[cat].append(dec)
    return new_annots

def get_demos(annots, cat):
    """
    Generate a formatted string of annotations excluding a specified category.

    Args:
        annots (dict): A dictionary where keys are categories and values are lists of annotations.
        cat (str): The category to exclude from the annotations.

    Returns:
        tuple: A tuple containing:
            - max_annot (str): The category with the maximum number of annotations (excluding the specified category).
            - demos (str): A formatted string of annotations from the category with the maximum annotations.
    """
    lens = {k: len(v) for k,v in annots.items() if k != cat}
    max_annot = max(lens.keys(), key=lens.get)
    lines = [f'* "{x}"' for x in annots[max_annot]]
    # lines = ['[START]'] + lines + ['[END]']
    demos = '\n'.join(lines)
    return max_annot, demos


def oneshot():
    """
    Processes a subset of test samples and generates output based on annotations.

    This function iterates over the first 10 test samples, loads their annotations,
    groups them, and saves them in a specified output directory. For each category
    from 1 to 9, it retrieves demonstration examples, resolves the source text,
    and generates a response using a one-shot prompting method. The responses are
    then saved in the output directory.

    Exceptions:
        torch.cuda.OutOfMemoryError: If a CUDA out-of-memory error occurs, it prints 'OOM' and continues with the next sample.

    Note:
        The function assumes that the following functions and variables are defined elsewhere in the code:
        - `test_samples`: A list of test sample filenames.
        - `data_dir`: The directory where the data files are located.
        - `group_annots(annots)`: A function that groups annotations.
        - `get_demos(annots, cat)`: A function that retrieves demonstration examples for a given category.
        - `resolve_src(fn)`: A function that resolves the source text for a given filename.
        - `prompt_oneshot(text, cat, demo_cat, demos)`: A function that generates a response using one-shot prompting.
    """
    for i, fn in enumerate(test_samples[:10]):
        print(i)
        try:
            annots = json.load(open(path.join(data_dir, 'data', fn)))[0]['annotations']
            annots = group_annots(annots)
            out_dir = path.join('gens', 'one', str(i))
            os.makedirs(out_dir, exist_ok=True)
            json.dump(annots, open(path.join(out_dir, 'labels.json'), 'w'))
            for cat in range(1, 10):
                demo_cat, demos = get_demos(annots, cat)
                text = resolve_src(fn)
                response = prompt_oneshot(text, cat, demo_cat, demos)
                # print(response)
                with open(path.join(out_dir, f'cat_{cat}'), 'w') as f:
                    f.write(response)
        except torch.cuda.OutOfMemoryError:
            print('OOM')
            continue

zeroshot()
# oneshot()
