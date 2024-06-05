import os
import json
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

data_dir = '/data/mohamed/data/mimic_decisions'

def get_diff(diff_type, fn, s, e):
    d = None
    c = 0
    while d is None and c < 100:
        try:
            d = diff.loc[fn, s, e][diff_type].item()
        except:
            e += 1
            c += 1
    if d is None:
        d = -1
    return d

def load_diff():
    df = pd.read_csv(os.path.join(data_dir, 'annotation_umls_count.csv'))
    df.set_index(['span_discharge_summary_id', 'start', 'end'], inplace=True)
    return df

def compute_diff():
    df = pd.read_csv(os.path.join(data_dir, 'annotation_umls_count.csv'))
    cats = np.array(df['categories'])
    discr = KBinsDiscretizer(3, encode='ordinal', strategy='quantile')
    diff_class_umls = discr.fit_transform(cats.reshape(-1,1)).reshape(-1)
    cats = np.array(df['len'])
    discr = KBinsDiscretizer(3, encode='ordinal', strategy='quantile')
    diff_class_len = discr.fit_transform(cats.reshape(-1,1)).reshape(-1)
    df['diff_umls'] = diff_class_umls
    df['diff_len'] = diff_class_len
    print(df)
    df.to_csv(os.path.join(data_dir, 'annotation_umls_count.csv'), index=False)


def fix_df():
    df = pd.read_csv(os.path.join(data_dir, 'annotation_umls_count.csv'))

    starts, ends, lens = [], [], []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        dec = row['decision']
        fn = row['span_discharge_summary_id']
        annots_file = glob(f'{os.path.join(data_dir, "data")}/*/{fn}*')[0]
        annots = json.load(open(annots_file), strict=False)[0]['annotations']
        dec_src = [x for x in annots if x['decision'] == dec][0]
        starts.append(dec_src['start_offset'])
        ends.append(dec_src['end_offset'])
        lens.append(len(dec))
    df['start'] = starts
    df['end'] = ends
    df['len'] = lens
    df.drop(['decision', 'category'], inplace=True, axis=1)
    df.to_csv(os.path.join(data_dir, 'annotation_umls_count.csv'), index=False)


diff = load_diff()

if __name__ == '__main__':
    # fix_df()
    # compute_diff()
    df = load_diff()
