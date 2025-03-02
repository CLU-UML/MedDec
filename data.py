import torch
import json
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from glob import glob
from collections.abc import Iterable
from collections import defaultdict


pheno_map = {'alcohol.abuse': 0,
        'advanced.lung.disease': 1,
        'advanced.heart.disease': 2,
        'chronic.pain.fibromyalgia': 3,
        'other.substance.abuse': 4,
        'psychiatric.disorders': 5,
        'obesity': 6,
        'depression': 7,
        'advanced.cancer': 8,
        'chronic.neurological.dystrophies': 9,
        'none': -1
        }
rev_pheno_map = {v: k for k,v in pheno_map.items()}
valid_cats = range(0,9)

def gen_splits(args, phenos):
    """
    Generates training, validation, and test splits for the dataset based on the provided arguments and phenotypes.

    Parameters:
    args (Namespace): A namespace object containing the following attributes:
        - unseen_pheno (str or None): The phenotype to be excluded from training and validation sets.
        - data_dir (str): The directory where the data is stored.
        - task (str): The task type, either 'token' or another task.
        - pheno_id (bool): A flag indicating whether to use phenotype IDs.
    phenos (DataFrame or None): A DataFrame containing phenotype information with columns 'subject_id' and 'phenotype_label'.

    Returns:
    tuple: A tuple containing three elements:
        - train (list or DataFrame): The training set, either as a list of file paths or a DataFrame of subjects.
        - val (list or DataFrame): The validation set, either as a list of file paths or a DataFrame of subjects.
        - test (list or DataFrame): The test set, either as a list of file paths or a DataFrame of subjects.

    Raises:
    ValueError: If phenos is None and args.task is not 'token'.
    """
    #if args.unseen_pheno is None:
    #    splits_dir = os.path.join(args.data_dir, 'splits')
    #    train_files = open(os.path.join(splits_dir, 'train.txt')).read().splitlines()
    #    val_files = open(os.path.join(splits_dir, 'val.txt')).read().splitlines()
    #    test_files = open(os.path.join(splits_dir, 'test.txt')).read().splitlines()
    #    return train_files, val_files, test_files

    np.random.seed(0)
    if args.task == 'token':
        #files = glob(os.path.join(args.data_dir, 'data/**/*'))
        files = glob(os.path.join(args.data_dir, 'data/*.json'))
        files = ["/".join(x.split('/')[-2:]) for x in files]
        subjects = np.unique([os.path.basename(x).split('_')[0] for x in files])
    elif phenos is not None:
        subjects = phenos['subject_id'].unique()
    else:
        raise ValueError

    phenos['phenotype_label'] = phenos['phenotype_label'].apply(lambda x: x.lower())

    n = len(subjects)
    print('Number of subjects:', n)
    train_count = int(0.8*n)
    val_count = max(0, int(0.9*n) - train_count)
    test_count = n - train_count - val_count

    train, val, test = [], [], []
    np.random.shuffle(subjects)
    subjects = list(subjects)
    pheno_list = set(np.unique(list(pheno_map.keys())).tolist())
    # pheno_list = set(pheno_map.keys())
    if args.unseen_pheno is not None:
        test_phenos = {rev_pheno_map[args.unseen_pheno]}
        unseen_pheno = rev_pheno_map[args.unseen_pheno]
        train_phenos = pheno_list - test_phenos
    else:
        test_phenos = pheno_list
        train_phenos = pheno_list
        unseen_pheno = 'null'
    while len(subjects) > 0:
        if len(pheno_list) > 0:
            for pheno in pheno_list:
                if len(train) < train_count and pheno in train_phenos:
                    el = None
                    for i, subj in enumerate(subjects):
                        row = phenos[phenos.subject_id == subj]
                        if row['phenotype_label'].apply(lambda x: pheno in x and not unseen_pheno in x).any():
                            el = subjects.pop(i)
                            break
                    if el is not None:
                        train.append(el)
                    elif el is None:
                        pheno_list.remove(pheno)
                        break
                if len(val) < val_count and (not args.pheno_id or len(val) <= (0.5*val_count)):
                    el = None
                    for i, subj in enumerate(subjects):
                        row = phenos[phenos.subject_id == subj]
                        if row['phenotype_label'].apply(lambda x: pheno in x).any():
                            el = subjects.pop(i)
                            break
                    if el is not None:
                        val.append(el)
                    elif el is None:
                        pheno_list.remove(pheno)
                        break
                if len(test) < test_count or (args.unseen_pheno is not None and pheno in test_phenos):
                    el = None
                    for i, subj in enumerate(subjects):
                        row = phenos[phenos.subject_id == subj]
                        if row['phenotype_label'].apply(lambda x: pheno in x).any():
                            el = subjects.pop(i)
                            break
                    if el is not None:
                        test.append(el)
                    elif el is None:
                        pheno_list.remove(pheno)
                        break
        else:
            if len(train) < train_count:
                el = subjects.pop()
                if el is not None:
                    train.append(el)
            if len(val) < val_count:
                el = subjects.pop()
                if el is not None:
                    val.append(el)
            if len(test) < test_count:
                el = subjects.pop()
                if el is not None:
                    test.append(el)

    if args.task == 'token':
        train = [x for x in files if os.path.basename(x).split('_')[0] in train]
        val = [x for x in files if os.path.basename(x).split('_')[0] in val]
        test = [x for x in files if os.path.basename(x).split('_')[0] in test]
    elif phenos is not None:
        train = phenos[phenos.subject_id.isin(train)]
        val = phenos[phenos.subject_id.isin(val)]
        test = phenos[phenos.subject_id.isin(test)]
    return train, val, test

class MyDataset(Dataset):
    def __init__(self, args, tokenizer, data_source, phenos, train = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        self.train = train
        self.pheno_ids = defaultdict(list)
        self.dec_ids = {k: [] for k in pheno_map.keys()}
        #self.meddec_stats = pd.read_csv(os.path.join(args.data_dir, 'stats.csv')).set_index(['SUBJECT_ID', 'HADM_ID', 'ROW_ID'])
        #elf.stats = defaultdict(list)

        if args.task == 'seq': # phenotype prediction
            for i, row in data_source.iterrows():
                sample = self.load_phenos(args, row, i)
                self.data.append(sample)
        else:
            for i, fn in enumerate(data_source):
                sample = self.load_decisions(args, fn, i, phenos)
                self.data.append(sample)

    def get_col(self, col):
        return np.array([x[col] for x in self.data])

    def load_phenos(self, args, row, idx):
        """
        Loads and processes phenotype data for a given row.

        Args:
            args: An object containing various configuration parameters.
            row: A dictionary containing data for a specific row, including 'subject_id', 'hadm_id', 'row_id', and 'phenotype_label'.
            idx: An index value (not used in the current implementation).

        Returns:
            tuple: A tuple containing:
                - input_ids: Tokenized input IDs from the text file.
                - labels: A numpy array of phenotype labels.
                - ids: Currently set to None.
        """
        txt_path = os.path.join(args.data_dir, f'raw_text/{row["subject_id"]}_{row["hadm_id"]}_{row["row_id"]}.txt')
        text = open(txt_path).read()
        encoding = self.tokenizer.encode_plus(text,
                truncation=args.truncate_train if self.train else args.truncate_eval)
        ids = None

        labels = np.zeros(args.num_phenos)

        sample_phenos = row['phenotype_label']
        if sample_phenos != 'none':
            for pheno in sample_phenos.split(','):
                labels[pheno_map[pheno.lower()]] = 1

        if args.pheno_id is not None:
            if args.pheno_id == -1:
                labels = [0.0 if any(labels) else 1.0]
            else:
                labels = [labels[args.pheno_id]]

        return encoding['input_ids'], labels, ids

    def load_decisions(self, args, fn, idx, phenos):
        """
        Load decision annotations and encode text data for a given file.
        Args:
            args (Namespace): Arguments containing configuration parameters.
            fn (str): Filename of the data file to load.
            idx (int): Index of the current sample.
            phenos (DataFrame): DataFrame containing phenotype labels.
        Returns:
            dict: A dictionary containing encoded input IDs, labels, token-to-character mapping, 
                  and additional information if not in training mode.
        Raises:
            ValueError: If encoding start or end positions are None.
        Notes:
            - The function reads the text data from a file, encodes it using a tokenizer, and processes annotations.
            - It supports different label encoding schemes: 'multiclass', 'bo', 'boe', and default.
            - If not in training mode, additional information such as spans, file name, and token mask is included in the results.
        """
        basename = os.path.splitext(os.path.basename(fn))[0]
        file_dir = os.path.join(args.data_dir, fn)

        sid, hadm, rid = map(int, basename.split('_')[:3])
        txt_path = os.path.join(args.data_dir, f'raw_text/{basename}.txt')
        text = open(txt_path).read()
        encoding = self.tokenizer.encode_plus(text,
                max_length=args.max_len,
                truncation=args.truncate_train if self.train else args.truncate_eval,
                padding = 'max_length',
                )
        if (sid, hadm, rid) in phenos.index:
            sample_phenos = phenos.loc[sid, hadm, rid]['phenotype_label']
            for pheno in sample_phenos.split(','):
                self.pheno_ids[pheno].append(idx)


        with open(file_dir) as f:
            data = json.load(f, strict=False)
        annots = data['annotations']
            
        if args.label_encoding == 'multiclass':
            labels = np.full(len(encoding['input_ids']), args.num_labels-1, dtype=int)
        else:
            labels = np.zeros((len(encoding['input_ids']), args.num_labels))
        if not self.train:
            token_mask = np.ones(len(encoding['input_ids']))
        all_spans = []
        for annot in annots:
            start = int(annot['start_offset'])

            enc_start = encoding.char_to_token(start)
            i = 1
            while enc_start is None and i < 10:
                enc_start = encoding.char_to_token(start+i)
                i += 1
            if i == 10:
                break

            end = int(annot['end_offset'])
            enc_end = encoding.char_to_token(end)
            j = 1
            while enc_end is None and j < 10:
                enc_end = encoding.char_to_token(end+j)
                j += 1
            if j == 10:
                enc_end = len(encoding.input_ids)

            if enc_end == enc_start:
                enc_end += 1

            if enc_start is None or enc_end is None:
                raise ValueError

            cat = parse_cat(annot['category'])
            if cat:
                cat -= 1
            if cat is None or cat not in valid_cats:
                if annot['category'] == 'TBD' and not self.train:
                    token_mask[enc_start:enc_end] = 0
                continue

            if args.label_encoding == 'multiclass':
                cat1 = cat * 2
                cat2 = cat * 2 + 1
                if not any([x in [2*y for y in range(args.num_labels//2)] for x in labels[enc_start:enc_end]]):
                    labels[enc_start] = cat1
                    if enc_end > enc_start + 1:
                        labels[enc_start+1:enc_end] = cat2
                if not self.train:
                    all_spans.append({'token_start': enc_start, 'token_end': enc_end-1, 'label': cat, 'text_start': start, 'text_end': end})
            elif args.label_encoding == 'bo':
                cat1 = cat * 2
                cat2 = cat * 2 + 1
                labels[enc_start, cat1] = 1
                labels[enc_start+1:enc_end, cat2] = 1
            elif args.label_encoding == 'boe':
                cat1 = cat * 3
                cat2 = cat * 3 + 1
                cat3 = cat * 3 + 2
                labels[enc_start, cat1] = 1
                labels[enc_start+1:enc_end-1, cat2] = 1
                labels[enc_end-1, cat3] = 1
            else:
                labels[enc_start:enc_end, cat] = 1

        #row = self.meddec_stats.loc[sid, hadm, rid]

        #self.stats['gender'].append(row.GENDER)
        #self.stats['ethnicity'].append(row.ETHNICITY)
        #self.stats['language'].append(row.LANGUAGE)

        results = {
                'input_ids': encoding['input_ids'],
                'labels': labels,
                't2c': encoding.token_to_chars,
                }
        if not self.train:
            results['all_spans'] = all_spans,
            results['file_name'] = fn
            results['token_mask'] = token_mask
        return results

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def parse_cat(cat):
    """
    Parses a string to extract the first numeric value.

    Args:
        cat (str): The input string to parse.

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


def load_phenos(args):
    """
    Load and preprocess phenotype data from a CSV file.

    Args:
        args: An object containing the attribute 'data_dir', which specifies the directory path where the 'phenos.csv' file is located.

    Returns:
        pandas.DataFrame: A DataFrame containing the preprocessed phenotype data with the following modifications:
            - The column 'Ham_ID' is renamed to 'HADM_ID'.
            - Rows with 'phenotype_label' equal to '?' are removed.
            - All column names are converted to lowercase.
    """
    phenos = pd.read_csv(os.path.join(args.data_dir, 'phenos.csv'))
    phenos.rename({'Ham_ID': 'HADM_ID'}, inplace=True, axis=1)
    phenos = phenos[phenos.phenotype_label != '?']
    phenos.rename(lambda k: k.lower(), inplace=True, axis = 1)
    return phenos

def downsample(dataset):
    """
    Downsamples the dataset to balance the classes.

    This function takes a dataset and downsamples the majority class to match the number of samples
    in the minority class. It assumes that the dataset has a 'data' attribute, which is a list of 
    samples, and that each sample is a tuple where the second element is a list containing the class 
    label at index 0.

    Parameters:
    dataset (object): An object with a 'data' attribute, which is a list of samples. Each sample is 
                      expected to be a tuple where the second element is a list containing the class 
                      label at index 0.

    Returns:
    None: The function modifies the dataset in place.
    """
    data = dataset.data
    class0 = [x for x in data if x[1][0] == 0]
    class1 = [x for x in data if x[1][0] == 1]

    if len(class0) > len(class1):
        class0 = resample(class0, replace=False, n_samples=len(class1), random_state=0)
    else:
        class1 = resample(class1, replace=False, n_samples=len(class0), random_state=0)
    dataset.data = class0 + class1

def upsample(dataset):
    """
    Upsamples the minority class in the given dataset to match the size of the majority class.

    Parameters:
    dataset (object): An object containing the dataset. It is expected to have a 'data' attribute,
                      which is a list of tuples where the second element is a list containing the class label.

    Returns:
    None: The function modifies the dataset in place by upsampling the minority class.
    """
    data = dataset.data
    class0 = [x for x in data if x[1][0] == 0]
    class1 = [x for x in data if x[1][0] == 1]

    if len(class0) > len(class1):
        class1 = resample(class1, replace=True, n_samples=len(class0), random_state=0)
    else:
        class0 = resample(class0, replace=True, n_samples=len(class1), random_state=0)
    dataset.data = class0 + class1

def load_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)

def load_data(args):
    """
    Load and preprocess data for training, validation, and testing.

    Args:
        args (Namespace): A namespace object containing various arguments and configurations.

    Returns:
        tuple: A tuple containing the following elements:
            - train_dataloader (DataLoader): DataLoader for the training dataset with segmented collation.
            - val_dataloader (DataLoader): DataLoader for the validation dataset with full collation.
            - test_dataloader (DataLoader): DataLoader for the test dataset with full collation.
            - train_ns (DataLoader): DataLoader for the training dataset with full collation and batch size of 1.

    The function performs the following steps:
        1. Defines two collation functions: `collate_segment` and `collate_full`.
        2. Loads the tokenizer and sets vocabulary size and maximum length.
        3. Loads phenotype data and generates train, validation, and test splits.
        4. Creates datasets for training, validation, and testing.
        5. Optionally resamples the training dataset based on the `args.resample` parameter.
        6. Prints the sizes of the train, validation, and test datasets.
        7. Creates DataLoaders for the train, validation, and test datasets with appropriate collation functions.
    """
    from sklearn.utils import resample
    def collate_segment(batch):
        """
        Collates a batch of data for segment processing.

        Args:
            batch (list of dict): A list of dictionaries where each dictionary contains the keys:
                - 'input_ids' (list): The input token IDs.
                - 'labels' (list): The labels corresponding to the input tokens.
                - 't2c' (any): Additional data associated with the batch.
                - 'ids' (optional, list): Optional IDs associated with the input tokens.

        Returns:
            dict: A dictionary containing the collated batch data with the keys:
                - 'input_ids' (torch.Tensor): Tensor of input token IDs.
                - 'labels' (torch.Tensor): Tensor of labels.
                - 'ids' (list or None): List of IDs if present in the input batch, otherwise None.
                - 'mask' (torch.Tensor): Tensor mask indicating valid token positions.
                - 't2c' (list): List of additional data associated with the batch.
        """
        xs = []
        ys = []
        t2cs = []
        has_ids = 'ids' in batch[0]
        if has_ids:
            idss = []
        else:
            ids = None
        masks = []
        for i in range(len(batch)):
            x = batch[i]['input_ids']
            y = batch[i]['labels']
            if has_ids:
                ids = batch[i]['ids']
            n = len(x)
            if n > args.max_len:
                start = np.random.randint(0, n - args.max_len + 1)
                x = x[start:start + args.max_len]
                if args.task == 'token':
                    y = y[start:start + args.max_len]
                if has_ids:
                    new_ids = []
                    ids = [x[start:start + args.max_len] for x in ids]
                    for subids in ids:
                        subids = [idx for idx, x in enumerate(subids) if x]
                        new_ids.append(subids)
                    all_ids = set([y for x in new_ids for y in x])
                    nones = set(range(args.max_len)) - all_ids
                    new_ids.append(list(nones))
                mask = [1] * args.max_len
            elif n < args.max_len:
                x = np.pad(x, (0, args.max_len - n))
                if args.task == 'token':
                    y = np.pad(y, ((0, args.max_len - n), (0, 0)))
                mask = [1] * n + [0] * (args.max_len - n)
            else:
                mask = [1] * n
            xs.append(x)
            ys.append(y)
            t2cs.append(batch[i]['t2c'])
            if has_ids:
                idss.append(new_ids)
            masks.append(mask)

        xs = torch.tensor(xs)
        ys = torch.tensor(ys)
        masks = torch.tensor(masks)
        return {'input_ids': xs, 'labels': ys, 'ids': ids, 'mask': masks, 't2c': t2cs}

    def collate_full(batch):
        """
        Collates a batch of data samples into a single batch suitable for model input.

        Args:
            batch (list of dict): A list of dictionaries where each dictionary represents a single data sample.
                Each dictionary should contain the keys 'input_ids', 'labels', and optionally 'all_spans' and 'file_name'.

        Returns:
            dict: A dictionary containing the collated batch data. The keys are:
                - 'input_ids': A tensor of padded input IDs.
                - 'labels': A tensor of padded labels.
                - 'mask': A tensor indicating the valid positions in the input IDs.
                - 'all_spans' (optional): A list of all spans from the batch.
                - 'file_name' (optional): A list of file names from the batch.
        """
        lens = [len(x['input_ids']) for x in batch]
        max_len = max(args.max_len, max(lens))
        for i in range(len(batch)):
            batch[i]['input_ids'] = np.pad(batch[i]['input_ids'], (0, max_len - lens[i]))
            if args.task == 'token':
                if args.label_encoding == 'multiclass':
                    batch[i]['labels'] = np.pad(batch[i]['labels'], (0, max_len - lens[i]), constant_values=-100)
                else:
                    batch[i]['labels'] = np.pad(batch[i]['labels'], ((0, max_len - lens[i]), (0, 0)))
            mask = [1] * lens[i] + [0] * (max_len - lens[i])
            batch[i]['mask'] = mask

        new_batch = {}
        for k in batch[0].keys():
            collated = [sample[k] for sample in batch]
            if k in ['all_spans', 'file_name']:
                new_batch[k] = collated
            elif isinstance(batch[0][k], Iterable):
                new_batch[k] = torch.tensor(np.array(collated))
            else:
                new_batch[k] = collated
        return new_batch

    tokenizer = load_tokenizer(args.model_name)
    args.vocab_size = tokenizer.vocab_size
    args.max_length = min(tokenizer.model_max_length, 512)

    phenos = load_phenos(args)
    train_files, val_files, test_files = gen_splits(args, phenos)
    phenos.set_index(['subject_id', 'hadm_id', 'row_id'], inplace=True)

    train_dataset = MyDataset(args, tokenizer, train_files, phenos, train=True)
    val_dataset = MyDataset(args, tokenizer, val_files, phenos)
    test_dataset = MyDataset(args, tokenizer, test_files, phenos)

    if args.resample == 'down':
        downsample(train_dataset)
    elif args.resample == 'up':
        upsample(train_dataset)

    print('Train dataset:', len(train_dataset))
    print('Val dataset:', len(val_dataset))
    print('Test dataset:', len(test_dataset))

    train_ns = DataLoader(train_dataset, 1, False,
            collate_fn=collate_full,
            )
    train_dataloader = DataLoader(train_dataset, args.batch_size, True,
            collate_fn=collate_segment,
            )
    val_dataloader = DataLoader(val_dataset, 1, False, collate_fn=collate_full)
    test_dataloader = DataLoader(test_dataset, 1, False, collate_fn=collate_full)

    return train_dataloader, val_dataloader, test_dataloader, train_ns
