# MedDec: A Dataset for Extracting Medical Decisions from Clinical Narratives

This is the code and dataset described in the [MedDec paper](/) (Findings of ACL: ACL 2024).

# Dataset

The dataset is made available through this link: [https://physionet.org/TBD](https://physionet.org/). 

The user must sign a data usage agreement before accessing the dataset.

### Phenotypes Annotations

The phenotype annotations used in the paper are available here: [https://physionet.org/content/phenotype-annotations-mimic/1.20.03/](https://physionet.org/content/phenotype-annotations-mimic/1.20.03/).

# Running the Baselines

```
python main.py --label_encoding multiclass --model_name google/electra-base-discriminator --total_steps 5000 --lr 4e-5
```

# Citation

TBA
