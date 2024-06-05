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
    for i,c in enumerate(cat):
        if c.isnumeric():
            if cat[i+1].isnumeric():
                return int(cat[i:i+2])
            return int(c)
    return None

def group_annots(annots):
    new_annots = defaultdict(list)
    for ann in annots:
        cat = parse_cat(ann['category'])
        if cat is None:
            continue
        dec = ann['decision']
        new_annots[cat].append(dec)
    return new_annots

def get_demos(annots, cat):
    lens = {k: len(v) for k,v in annots.items() if k != cat}
    max_annot = max(lens.keys(), key=lens.get)
    lines = [f'* "{x}"' for x in annots[max_annot]]
    # lines = ['[START]'] + lines + ['[END]']
    demos = '\n'.join(lines)
    return max_annot, demos


def oneshot():
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
