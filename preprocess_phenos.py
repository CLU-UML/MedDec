# Aggregate the annotations of the same SUBJECT_ID, HADM_ID, ROW_ID into a single phenotype label.
# If there is only one annotation, use that as the gold data.
# If there are multiple annotations, prioritize annotators based on the following order:
# DAG or PAT are prioritized over all other annotators.
# JF or JTW are prioritized over ETM and JW.
# If there are multiple annotations from the same prioritized annotator, use the sum of the annotations.
# If one of the annotations is NONE or UNSURE, the phenotype label is '?'.

import pandas as pd
import os, sys

def aggregate_annotations(df):

    def process_group(group):
        if len(group) == 1:
            return process_single_annotation(group.iloc[0])
        else:
            return process_multiple_annotations(group)

    def process_single_annotation(row):
        """
        Processes a single annotation row to generate a phenotype label.

        Args:
            row (pd.Series): A pandas Series object representing a single row of data.

        Returns:
            pd.Series: A pandas Series object with a single key 'phenotype_label' containing a 
                       comma-separated string of phenotype labels. If no phenotype labels are 
                       found, returns '?'.
        """
        phenotype_label = [col if col != 'UNSURE' else '?' for col in PHENOTYPE_COLUMNS if row[col] > 0]
        return pd.Series({'phenotype_label': ','.join(phenotype_label) if phenotype_label else '?'})

    def process_multiple_annotations(group):
        """
        Processes a group of annotations by prioritizing certain operators and summarizing phenotype labels.
        Args:
            group (pd.DataFrame): A DataFrame containing annotation data with columns 'BATCH.ID', 'OPERATOR', 
                and phenotype columns.
        Returns:
            pd.Series: A Series with the summarized phenotype label and concatenated operator names.
            
        The function performs the following steps:
        1. Prioritizes annotations based on predefined operator groups.
        2. Drops duplicate rows based on all columns except 'BATCH.ID' and 'OPERATOR'.
        3. If there are multiple unique rows and any of them have a 'NONE' phenotype, returns a '?' phenotype label.
        4. Sums over phenotype columns and concatenates operator names.
        5. Generates a phenotype label based on the summed phenotype columns, replacing 'UNSURE' with '?'.
        """
        priority_operators = [['DAG', 'PAT'], ['JTW', 'JF'], ['ETM', 'JW']]
        for operator_group in priority_operators:
            if group['OPERATOR'].isin(operator_group).any():
                selected_rows = group[group['OPERATOR'].isin(operator_group)]
                break
        else:
            selected_rows = group

        selected_rows_unique = selected_rows.drop(['BATCH.ID', 'OPERATOR'], axis=1).drop_duplicates()
        
        if len(selected_rows_unique) > 1 and selected_rows_unique['NONE'].sum() > 0:
            return pd.Series({
                'phenotype_label': '?',
                'OPERATOR': ','.join(selected_rows['OPERATOR'].unique())
            })
        
        # Sum over phenotype_columns and keep other unchanged
        selected_rows_unique = selected_rows_unique.sum()
        selected_rows_unique['OPERATOR'] = ','.join(selected_rows['OPERATOR'].unique())

        # selected_rows = selected_rows_unique.sum()
        phenotype_label = [col if col != 'UNSURE' else '?' for col in PHENOTYPE_COLUMNS if selected_rows_unique[col] > 0]
        return pd.Series({'phenotype_label': ','.join(sorted(phenotype_label)) if phenotype_label else '?',
                'OPERATOR': ",".join(selected_rows['OPERATOR'].unique())})

    return df.groupby(['SUBJECT_ID', 'HADM_ID', 'ROW_ID']).apply(process_group).reset_index()




# Constants
PHENOTYPE_COLUMNS = ['ADVANCED.CANCER', 'ADVANCED.HEART.DISEASE', 'ADVANCED.LUNG.DISEASE', 'ALCOHOL.ABUSE', 
                     'CHRONIC.NEUROLOGICAL.DYSTROPHIES', 'CHRONIC.PAIN.FIBROMYALGIA', 'DEPRESSION', 'OBESITY', 
                     'OTHER.SUBSTANCE.ABUSE', 'PSYCHIATRIC.DISORDERS', 'NONE']

# Paths
if len(sys.argv) != 2:
    print('Usage: python preprocess_phenos.py <input_file>')
    sys.exit(1)
INPUT_FILE = sys.argv[1]
OUTPUT_FILE = os.path.join(os.path.dirname(INPUT_FILE), 'phenos.csv')

# Main execution
if __name__ == "__main__":
    # Read and preprocess data
    df = pd.read_csv(INPUT_FILE)
    df['PSYCHIATRIC.DISORDERS'] = df['DEMENTIA'] | df['DEVELOPMENTAL.DELAY.RETARDATION'] | df['SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS']
    df = df[['SUBJECT_ID', 'HADM_ID', 'ROW_ID'] + PHENOTYPE_COLUMNS + ['OPERATOR', 'BATCH.ID']]

    # Aggregate annotations
    result_df = aggregate_annotations(df)
    result_df.to_csv(OUTPUT_FILE, index=False)
