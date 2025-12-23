# MedDec: A Dataset for Extracting Medical Decisions from Discharge Summaries

![MedDec](assets/figure.png)

This is the code and dataset described in **[MedDec (Elgaar et al., Findings of ACL: ACL 2024)](https://aclanthology.org/2024.findings-acl.975/)**.

MedDec is the first dataset specifically developed for extracting and classifying medical decisions from clinical notes. It includes 451 expert-annotated annotated discharge summaries from the MIMIC-III dataset, offering a valuable resource for understanding and facilitating clinical decision-making.

# Dataset

> [!TIP]
> The dataset has been released as of October 16, 2024.

The dataset is available through this link: **[https://physionet.org/content/meddec/1.0.0/](https://physionet.org/content/meddec/1.0.0/)**. The user must sign a data usage agreement before accessing the dataset.

### Phenotypes Annotations

The phenotype annotations used in the paper are available here: [https://physionet.org/content/phenotype-annotations-mimic/1.20.03/](https://physionet.org/content/phenotype-annotations-mimic/1.20.03/).

# Prerequisites

### Requirements

Install the required packages using the following command:
```
pip install -r requirements.txt
```

> [!NOTE]
> This repository requires **Python 3**. If your system does not have `python` / `pip` pointing to Python 3, use `python3` / `pip3` in the commands below.

### Extract Notes Text from MIMIC-III

To extract the notes text from the MIMIC-III dataset, run the following command:
```
python extract_texts.py <data_dir> <notes_path (NOTEEVENTS.csv)>
```
- `data_dir`: Directory where the dataset is stored.
- `notes_path`: Path to the `NOTEEVENTS.csv` file. The texts will be written to the `data_dir/raw_text` directory.

### Aggregate Phenotype Annotations

To preprocess the phenotype annotations, run the following command:
```
python preprocess_phenos.py <phenotypes_path (ACTdb102003.csv)>
```
- `phenotypes_path`: Path to the phenotype annotations. The aggregated annotations will be written to `phenos.csv` in the same directory as the input file.

# Running the Baselines

The code expects `data`, `raw_text`, and `phenos.csv` to be in the `data_dir` directory.

To train the baselines, run the following command:
```
python main.py --data_dir <data_dir> --label_encoding multiclass --model_name google/electra-base-discriminator --total_steps 5000 --lr 4e-5
```

To evaluate the baselines, run the following command:
```
python main.py --data_dir <data_dir> --eval_only --ckpt ./checkpoints/[datetime]-[model_name]
```

## Arguments

- `data_dir`: The directory where the dataset is stored. The default is `./data/`.
- `pheno_path`: The path to the phenotype annotations. The default is `./ACTdb102003.csv`.
- `task`: `token` is the token classification task (decision extraction), and `seq` is the sequence classification task (phenotype prediction). The default is `token`.
- `eval_only`: Whether to evaluate the model only. `--ckpt` should be provided. The default is `False`.
- `label_encoding`: `multiclass`, `bo` (beginning inside outside), or `boe` (beginning outside end). The default is `multiclass`.
- `truncate_train`: Truncate the training sequences to a maximum length. Otherwise, the sequences are randomly chunked at training time. The default is `False`.
- `truncate_eval`: Truncate the evaluation sequences to a maximum length. The default is `False`.
- `use_crf`: Whether to use a CRF layer. The default is `False`.
- `model_name`: The name of the model from Hugging Face Transformers
- `total_steps`: The number of training steps
- `lr`: The learning rate
- `batch_size`: The batch size
- `seed`: The random seed

# Shared Task

This branch includes helper scripts and files to support the MedDec shared task: official split files, an example submission file, and a standalone evaluator.

## Official splits

- `splits/train.txt`: 350 discharge summaries (one ID per line)
- `splits/val.txt`: 53 discharge summaries (one ID per line)

Each line is a MedDec discharge summary ID of the form:
`[SUBJECT_ID]_[HADM_ID]_[ROW_ID]` (no file extension).

> [!NOTE]
> The official **test split is not publicly released** during the shared task.

## Submission / predictions file format

The evaluator expects a JSON **list** with this schema:
```json
[
  {
    "file_name": "78_100536_1787",
    "predictions": [
      { "start_offset": 322, "end_offset": 332, "category": 3 }
    ]
  }
]
```

- `file_name`: discharge summary ID (matches the `.json`/`.txt` basename)
- `predictions`: list of predicted spans
- `start_offset`, `end_offset`: character offsets into the raw discharge summary text (0-based; `[start_offset, end_offset)` )
- `category`: integer decision category ID (1–9)

An example file in the correct format is provided in `predictions_validation.json` (useful for sanity-checking your setup).

## `evaluate.py` (official evaluator)

Computes:
- **Span F1**: label + text match after light span boundary normalization
- **Token F1**: token-level macro-F1 per note (then averaged across notes)
- **Shared-task competition score** (only when `--stats_csv` is provided):
  - **base_score**: overall performance, defined as the average of overall Span F1 and overall Token F1.
  - **worst_group_score**: the *lowest* score among all evaluated subgroups (sex, race, language), where each subgroup score is computed the same way as `base_score` but restricted to the notes in that subgroup.  
  - **final_score**: the competition score, defined as the average of `base_score` and `worst_subgroup_score`. This rewards strong overall performance *and* robustness across subgroups.

Shared-task score definition:
```
base_score            = (overall_span_f1 + overall_token_f1) / 2
subgroup_score(g)     = (span_f1_on_group_g + token_f1_on_group_g) / 2
worst_subgroup_score  = min_g subgroup_score(g)
final_score           = (base_score + worst_subgroup_score) / 2
```

Subgroups `g` are defined using `stats.csv` over:
- sex: Female, Male
- race: White, African American, Hispanic, Asian, Other
- language: English, Non-English

> [!IMPORTANT]
> Offsets must be computed against the **exact raw text** used by the evaluator (`--raw_text_dir/<file_name>.txt`). Any text normalization (whitespace cleanup, newline conversion, section filtering, etc.) will shift offsets and hurt scoring.

Basic usage (evaluate on validation split):
```
python evaluate.py \
  --gold_dir MedDec/data \
  --raw_text_dir MedDec/raw_text \
  --predictions predictions_validation.json \
  --split_file splits/val.txt
```

Optional: save results to a JSON file:
```
python evaluate.py \
  --gold_dir MedDec/data \
  --raw_text_dir MedDec/raw_text \
  --predictions predictions_validation.json \
  --split_file splits/val.txt \
  --output results.json
```

Optional: shared-task final score with subgroup (worst-group) adjustment (requires `stats.csv`, see below):
```
python evaluate.py \
  --gold_dir MedDec/data \
  --raw_text_dir MedDec/raw_text \
  --predictions predictions_validation.json \
  --split_file splits/val.txt \
  --stats_csv MedDec/stats.csv
```

## `generate_stats.py` (build `stats.csv` for subgroup scoring)

`evaluate.py --stats_csv ...` uses a `stats.csv` file keyed by `(SUBJECT_ID, HADM_ID, ROW_ID)` to compute subgroup scores for:
- sex (Female/Male)
- race (White/African American/Hispanic/Asian/Other)
- language (English/Non-English)

To generate `stats.csv` you need local access to MIMIC-III tables `ADMISSIONS` and `PATIENTS` (either `.csv` or `.csv.gz`):
```
python generate_stats.py <meddec_dir> <mimic_dir> [output_path]
```

Example:
```
python generate_stats.py MedDec /path/to/mimic-iii-1.4 MedDec/stats.csv
```

## `clean_data.py` (optional: normalize gold span boundaries)

This script rewrites MedDec annotation offsets to better align with word boundaries and strips leading/trailing punctuation. It:
- moves `MedDec/data` → `MedDec/data_unclean` (one-time)
- writes cleaned JSON files back into `MedDec/data`

Run:
```
python clean_data.py --data_dir MedDec
```



# Citation

If you use this dataset or code, please consider citing the following paper:
```bibtex
@inproceedings{elgaar-etal-2024-meddec,
    title = "{M}ed{D}ec: A Dataset for Extracting Medical Decisions from Discharge Summaries",
    author = "Elgaar, Mohamed and Cheng, Jiali and Vakil, Nidhi and Amiri, Hadi and Celi, Leo Anthony",
    editor = "Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.975",
    pages = "16442--16455",
}
```

Additionally, please cite the dataset as follows:
```bibtex
@misc{elgaar2024meddec,
    title = "MedDec: Medical Decisions for Discharge Summaries in the MIMIC-III Database",
    author = "Elgaar, Mohamed and Cheng, Jiali and Vakil, Nidhi and Amiri, Hadi and Celi, Leo Anthony",
    year = "2024",
    version = "1.0.0",
    publisher = "PhysioNet",
    url = "https://doi.org/10.13026/nqnw-7d62"
}
```
