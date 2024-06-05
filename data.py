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
    np.random.seed(0)
    if args.task == 'token':
        files = glob(os.path.join(args.data_dir, 'data/**/*'))
        files = ["/".join(x.split('/')[-2:]) for x in files]
        subjects = np.unique([os.path.basename(x).split('_')[0] for x in files])
    elif phenos is not None:
        subjects = phenos['subject_id'].unique()
    else:
        raise ValueError

    phenos['phenotype_label'] = phenos['phenotype_label'].apply(lambda x: x.lower())

    n = len(subjects)
    train_count = int(0.8*n)
    # train_count = n
    val_count = max(0, int(0.9*n) - train_count)
    test_count = n - train_count - val_count

    train, val, test = [], [], []
    np.random.shuffle(subjects)
    subjects = list(subjects)
    pheno_list = np.unique(list(pheno_map.keys())).tolist()
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
        self.meddec_stats = pd.read_csv(os.path.join(args.data_dir, 'stats.csv')).set_index(['SUBJECT_ID', 'HADM_ID', 'ROW_ID'])
        self.stats = defaultdict(list)

        if args.task == 'seq':
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
        txt_candidates = glob(os.path.join(args.data_dir,
            f'raw_text/{row["subject_id"]}_{row["hadm_id"]}*.txt'))
        text = open(txt_candidates[0]).read()
        if args.pheno_n == 500:

            file_dir = glob(os.path.join(args.data_dir,
                f'data/*/{row["subject_id"]}_{row["hadm_id"]}*.json'))[0]
            with open(file_dir) as f:
                data = json.load(f, strict=False)
            annots = data[0]['annotations']

            encoding = self.tokenizer.encode_plus(text,
                    truncation=args.truncate_train if self.train else args.truncate_eval)

            ids = np.zeros((args.num_decs, len(encoding['input_ids'])))
            for annot in annots:
                start = int(annot['start_offset'])

                enc_start = encoding.char_to_token(start)
                i = 1
                while enc_start is None:
                    enc_start = encoding.char_to_token(start+i)
                    i += 1

                end = int(annot['end_offset'])
                enc_end = encoding.char_to_token(end)
                j = 1
                while enc_end is None:
                    enc_end = encoding.char_to_token(end-j)
                    j += 1

                if enc_start is None or enc_end is None:
                    raise ValueError

                cat = parse_cat(annot['category'])
                if not cat or cat not in valid_cats:
                    continue
                ids[cat-1, enc_start:enc_end] = 1
        else:
            encoding = self.tokenizer.encode_plus(text,
                    truncation=args.truncate_train if self.train else args.truncate_eval)
            ids = None

        labels = np.zeros(args.num_phenos)

        if args.pheno_n in (500, 800):
            sample_phenos = row['phenotype_label']
            if sample_phenos != 'none':
                for pheno in sample_phenos.split(','):
                    labels[pheno_map[pheno.lower()]] = 1

        elif args.pheno_n == 1500:
            for k,v in pheno_map.items():
                if row[k] == 1:
                    labels[v] = 1

        if args.pheno_id is not None:
            if args.pheno_id == -1:
                labels = [0.0 if any(labels) else 1.0]
            else:
                labels = [labels[args.pheno_id]]

        return encoding['input_ids'], labels, ids

    def load_decisions(self, args, fn, idx, phenos):
        basename = os.path.basename(fn).split("-")[0]
        file_dir = os.path.join(args.data_dir, 'data', fn)

        pheno_id = "_".join(basename.split("_")[:3]) + '.txt'
        txt_candidates = glob(os.path.join(args.data_dir,
            f'raw_text/{basename}*.txt'))
        text = open(txt_candidates[0]).read()
        encoding = self.tokenizer.encode_plus(text,
                max_length=args.max_len,
                truncation=args.truncate_train if self.train else args.truncate_eval,
                padding = 'max_length',
                )
        if pheno_id in phenos.index:
            sample_phenos = phenos.loc[pheno_id]['phenotype_label']
            for pheno in sample_phenos.split(','):
                self.pheno_ids[pheno].append(idx)


        with open(file_dir) as f:
            data = json.load(f, strict=False)
        annots = data[0]['annotations']
            
        if args.label_encoding == 'multiclass':
            labels = np.full(len(encoding['input_ids']), args.num_labels-1, dtype=int)
        else:
            labels = np.zeros((len(encoding['input_ids']), args.num_labels))
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
                continue

            if args.label_encoding == 'multiclass':
                cat1 = cat * 2
                cat2 = cat * 2 + 1
                if not any([x in [2*y for y in range(args.num_labels//2)] for x in labels[enc_start:enc_end]]):
                    labels[enc_start] = cat1
                    if enc_end > enc_start + 1:
                        labels[enc_start+1:enc_end] = cat2
                if not self.train:
                    all_spans.append({'token_start': enc_start, 'token_end': enc_end, 'label': cat, 'text_start': start, 'text_end': end})
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

        sid, hadm, rid = map(int, basename.split('_')[:3])
        row = meddec_stats.loc[sid, hadm, rid]

        self.stats['gender'].append(row.GENDER)
        self.stats['ethnicity'].append(row.ETHNICITY)
        self.stats['language'].append(row.LANGUAGE)

        results = {
                'input_ids': encoding['input_ids'],
                'labels': labels,
                't2c': encoding.token_to_chars,
                }
        if not self.train:
            results['all_spans'] = all_spans,
            results['file_name'] = fn
        return results

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def parse_cat(cat):
    for i,c in enumerate(cat):
        if c.isnumeric():
            if cat[i+1].isnumeric():
                return int(cat[i:i+2])
            return int(c)
    return None


def load_phenos(args):
    if args.pheno_n == 500:
        phenos = pd.read_csv(args.pheno_path, sep='\t').rename(lambda x: x.strip(), axis=1)
        phenos['raw_text'] = phenos['raw_text'].apply(lambda x: os.path.basename(x))
        phenos[['SUBJECT_ID', 'HADM_ID', 'ROW_ID']] = \
            [os.path.splitext(x)[0].split('_')[:3] for x in phenos['raw_text']]
        phenos = phenos[phenos['phenotype_label'] != '?']
    elif args.pheno_n == 800:
        phenos = pd.read_csv(args.pheno_path)
        phenos.rename({'Ham_ID': 'HADM_ID'}, inplace=True, axis=1)
        phenos = phenos[phenos.phenotype_label != '?']
    elif args.pheno_n == 1500:
        phenos = pd.read_csv(args.pheno_path)
        phenos.rename({'Hospital.Admission.ID': 'HADM_ID',
            'subject.id': 'SUBJECT_ID'}, inplace=True, axis=1)
        phenos = phenos[phenos.Unsure != 1]
        phenos['psychiatric.disorders'] = phenos['Dementia']\
                                        | phenos['Developmental.Delay.Retardation']\
                                        | phenos['Schizophrenia.and.other.Psychiatric.Disorders']
    else:
        raise NotImplementedError("Please provide the phenotypes file")
    phenos.rename(lambda k: k.lower(), inplace=True, axis = 1)
    return phenos

def downsample(dataset):
    data = dataset.data
    class0 = [x for x in data if x[1][0] == 0]
    class1 = [x for x in data if x[1][0] == 1]

    if len(class0) > len(class1):
        class0 = resample(class0, replace=False, n_samples=len(class1), random_state=0)
    else:
        class1 = resample(class1, replace=False, n_samples=len(class0), random_state=0)
    dataset.data = class0 + class1

def upsample(dataset):
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
    from sklearn.utils import resample
    def collate_segment(batch):
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
    phenos.set_index('raw_text', inplace=True)

    train_dataset = MyDataset(args, tokenizer, train_files, phenos, train=True)
    val_dataset = MyDataset(args, tokenizer, val_files, phenos)
    test_dataset = MyDataset(args, tokenizer, test_files, phenos)
    # test_dataset = MyDataset(args, tokenizer, train_files, phenos)

    # import json
    # d = json.load(open('token_losses.json'))
    # j = 0
    # for i in range(len(test_dataset)):
    #     tokens = tokenizer.convert_ids_to_tokens(test_dataset[i]['input_ids'])
    #     spans = test_dataset[i]['all_spans'][0]
    #     fn = test_dataset[i]['file_name']
    #     for span in spans:
    #         start = span['token_start']
    #         end = span['token_end']
    #         d[j]['tokens'] = tokens[start:end]
    #         d[j]['file_name'] = fn
    #         d[j]['span'] = span
    #         j += 1
    # json.dump(d, open('token_losses.json', 'w'))
    # exit()

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

    train_files = [os.path.basename(x).split('-')[0] for x in train_files]
    val_files = [os.path.basename(x).split('-')[0] for x in val_files]
    test_files = [os.path.basename(x).split('-')[0] for x in test_files]

    return train_dataloader, val_dataloader, test_dataloader, train_ns, [train_files, val_files, test_files]
