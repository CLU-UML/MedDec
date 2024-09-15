# MedDec: A Dataset for Extracting Medical Decisions from Discharge Summaries

![MedDec](assets/figure.png)

This is the code and dataset described in **[MedDec (Elgaar et al., Findings of ACL: ACL 2024)](https://aclanthology.org/2024.findings-acl.975/)**.

MedDec is the first dataset specifically developed for extracting and classifying medical decisions from clinical notes. It includes 451 expert-annotated annotated discharge summaries from the MIMIC-III dataset, offering a valuable resource for understanding and facilitating clinical decision-making.

# Dataset

> [!NOTE]
> Currently under review on PhysioNet. Please check back on October 1, 2024.

The dataset is made available through this link: **[https://physionet.org/TBD](https://physionet.org/)**.

The user must sign a data usage agreement before accessing the dataset.

### Phenotypes Annotations

The phenotype annotations used in the paper are available here: [https://physionet.org/content/phenotype-annotations-mimic/1.20.03/](https://physionet.org/content/phenotype-annotations-mimic/1.20.03/).

# Prerequisites

### Requirements

Install the required packages using the following command:
```
pip install -r requirements.txt
```

### Extract Notes Text from MIMIC-III

To extract the notes text from the MIMIC-III dataset, run the following command:
```
python extract_notes.py <data_dir> <notes_path (NOTEEVENTS.csv)>
```
The `data_dir` is the directory where the dataset is stored, and the `notes_path` is the path to the `NOTEEVENTS.csv` file. The texts will be written to the `data_dir/raw_text` directory.


### Aggregate Phenotype Annotations

To preprocess the phenotype annotations, run the following command:
```
python preprocess_phenotypes.py <phenotypes_path (ACTdb102003.csv)>
```
This script aggregates the multiple annotations per row from the phenotype annotations to a single label.

The `phenotypes_path` is the path to the phenotype annotations. The aggregated annotations will be written to `phenos.csv` in the same directory as the input file.

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



# Citation

If you use this dataset or code, please cite the following paper:

```
@inproceedings{elgaar-etal-2024-meddec,
    title = "{M}ed{D}ec: A Dataset for Extracting Medical Decisions from Discharge Summaries",
    author = "Elgaar, Mohamed  and
      Cheng, Jiali  and
      Vakil, Nidhi  and
      Amiri, Hadi  and
      Celi, Leo Anthony",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.975",
    pages = "16442--16455",
}
```
