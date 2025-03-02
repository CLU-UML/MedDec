from glob import glob
import pandas as pd
import os, sys

if len(sys.argv) != 3:
    print('Usage: python extract_texts.py <data_dir> <notes_path>')
    sys.exit(1)

data_dir = sys.argv[1]
notes_path = sys.argv[2]
text_dir = os.path.join(data_dir, 'raw_text')

files = glob(os.path.join(data_dir, 'data/*.json'))
print('Number of files found:', len(files))
notes = pd.read_csv(notes_path).set_index(['SUBJECT_ID', 'HADM_ID', 'ROW_ID'])
print('Number of notes found:', len(notes))

os.makedirs(text_dir, exist_ok=True)
for fn in files:
    sid, hadm, rid = map(int, os.path.splitext(os.path.basename(fn))[0].split('_'))
    note = notes.loc[sid, hadm, rid]
    out_fn = f'{sid}_{hadm}_{rid}.txt'
    with open(os.path.join(text_dir, out_fn), 'w') as f:
        f.write(note.TEXT)

# Run this command: python extract_texts.py "data_dir/" "data_dir/NOTEEVENTS.csv"
