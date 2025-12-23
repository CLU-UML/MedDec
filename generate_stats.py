import pandas as pd
import os
import sys
from glob import glob

def get_ids_from_filename(filename):
    basename = os.path.basename(filename)
    name_no_ext = os.path.splitext(basename)[0]
    parts = name_no_ext.split('_')
    
    if len(parts) < 3:
        return None
        
    sid = int(parts[0])
    hadm = int(parts[1])
    
    # Handle RID which might have suffix
    rid_part = parts[2]
    # Split by '-' to remove annotator suffix if present
    rid = int(rid_part.split('-')[0])
    
    return sid, hadm, rid

def map_ethnicity(ethnicity):
    if pd.isna(ethnicity):
        return 'UNKNOWN/NOT SPECIFIED'
    
    ethnicity = str(ethnicity).upper()
    
    if 'WHITE' in ethnicity:
        return 'W'
    elif 'ASIAN' in ethnicity:
        return 'Asian'
    elif 'HISPANIC' in ethnicity or 'LATINO' in ethnicity:
        return 'Hisp.'
    elif 'BLACK' in ethnicity or 'AFRICAN' in ethnicity:
        return 'AA'
    elif 'NATIVE HAWAIIAN' in ethnicity or 'PACIFIC ISLANDER' in ethnicity:
        return 'NH'
    elif 'UNKNOWN' in ethnicity or 'NOT SPECIFIED' in ethnicity:
        return 'UNKNOWN/NOT SPECIFIED'
    elif 'UNABLE TO OBTAIN' in ethnicity:
        return 'UNABLE TO OBTAIN'
    elif 'PATIENT DECLINED TO ANSWER' in ethnicity:
        return 'PATIENT DECLINED TO ANSWER'
    else:
        return 'OTHER'

def map_language(language):
    if pd.isna(language):
        return 'Non-ENGL'
    if str(language).upper() == 'ENGL':
        return 'ENGL'
    return 'Non-ENGL'

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_stats.py <meddec_dir> <mimic_dir> [output_path]")
        sys.exit(1)

    meddec_dir = sys.argv[1]
    mimic_dir = sys.argv[2]
    
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    else:
        output_path = os.path.join(meddec_dir, 'stats.csv')

    print(f"MedDec Dir: {meddec_dir}")
    print(f"MIMIC Dir: {mimic_dir}")
    print(f"Output Path: {output_path}")

    # Find JSON files
    # Recursive search in meddec_dir/data
    search_pattern = os.path.join(meddec_dir, 'data', '**', '*.json')
    json_files = glob(search_pattern, recursive=True)
    
    print(f"Found {len(json_files)} JSON files")
    
    data = []
    for f in json_files:
        try:
            ids = get_ids_from_filename(f)
            if ids:
                data.append(ids)
        except Exception as e:
            print(f"Error parsing {f}: {e}")
            
    df = pd.DataFrame(data, columns=['SUBJECT_ID', 'HADM_ID', 'ROW_ID'])
    
    if df.empty:
        print("No valid JSON files found or parsed.")
        sys.exit(0)

    # Load MIMIC data
    print("Loading MIMIC tables...")
    admissions_path = os.path.join(mimic_dir, 'ADMISSIONS.csv.gz')
    patients_path = os.path.join(mimic_dir, 'PATIENTS.csv.gz')
    
    # Handle both .csv and .csv.gz
    if not os.path.exists(admissions_path):
        admissions_path = os.path.join(mimic_dir, 'ADMISSIONS.csv')
    if not os.path.exists(patients_path):
        patients_path = os.path.join(mimic_dir, 'PATIENTS.csv')

    print(f"Reading {admissions_path}")
    admissions = pd.read_csv(admissions_path, usecols=['SUBJECT_ID', 'HADM_ID', 'ETHNICITY', 'LANGUAGE'])
    print(f"Reading {patients_path}")
    patients = pd.read_csv(patients_path, usecols=['SUBJECT_ID', 'GENDER'])
    
    # Merge
    print("Merging data...")
    # Merge with patients first
    merged = df.merge(patients, on='SUBJECT_ID', how='left')
    
    # Merge with admissions
    merged = merged.merge(admissions, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    
    # Apply mappings
    merged['ETHNICITY'] = merged['ETHNICITY'].apply(map_ethnicity)
    merged['LANGUAGE'] = merged['LANGUAGE'].apply(map_language)
    
    # Select and reorder columns
    cols = ['SUBJECT_ID', 'HADM_ID', 'ROW_ID', 'ETHNICITY', 'LANGUAGE', 'GENDER']
    final_df = merged[cols]
    
    # Save
    print(f"Saving to {output_path}")
    final_df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == '__main__':
    main()
