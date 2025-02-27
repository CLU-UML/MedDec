import json
import os
from os import path

# Non-exact span-match (plus-minus 50 chars)
def get_tp(ys, preds, method='em'):
    """
    Calculate the number of true positives based on the given method.

    Args:
        ys (list): A list of true labels.
        preds (list): A list of predicted labels.
        method (str, optional): The method to use for comparison. 
                                'em' for exact match, 'approx-m' for approximate match. 
                                Defaults to 'em'.

    Returns:
        int: The count of true positives.
    """
    c = 0
    for pred in all_preds:
        pred_sample, pred_cat, pred_dec = pred
        for sample, cat, dec in all_labels:
            if method == 'em':
                if dec == pred_dec \
                        and pred_cat == cat \
                        and pred_sample == sample:
                    c+= 1
            elif method == 'approx-m':
                if (dec in pred_dec or pred_dec in dec) \
                        and pred_cat == cat \
                        and pred_sample == sample \
                        and abs(len(pred_dec.split()) - len(dec.split())) <= 10:
                    c+= 1
    return c


def f1_score(ys, preds, method='em'):
    """
    Calculate the F1 score for the given true labels and predicted labels.

    Parameters:
    ys (set): A set of true labels.
    preds (set): A set of predicted labels.
    method (str): The method to use for calculating true positives. Default is 'em'.

    Returns:
    float: The F1 score as a percentage.
    """
    # tp = len(preds & ys)
    tp = get_tp(ys, preds, method)
    fn = len(ys) - tp
    fp = len(preds) - tp
    f1 = (2 * tp / (2 * tp + fp + fn)) * 100 if tp + fp + fn > 0 else 0
    return f1

def process_labels(labels, sample):
    """
    Processes the given labels and sample to generate a set of tuples.

    Args:
        labels (dict): A dictionary where keys are categories (as strings) and values are lists of decisions (as strings).
        sample (str): A sample identifier.

    Returns:
        set: A set of tuples, each containing the sample identifier, category (as an integer), and a stripped decision string.
    """
    output = set()
    for cat, decs in labels.items():
        for dec in decs:
            output.add((sample, int(cat), dec.strip()))
    return output

def process_preds(preds, sample, cat):
    """
    Processes predictions by associating each prediction with a sample and category.

    Args:
        preds (list of str): A list of prediction strings.
        sample (str): The sample identifier.
        cat (str): The category identifier.

    Returns:
        set: A set of tuples, each containing the sample, category, and a stripped prediction.
    """
    output = set()
    for pred in preds:
        output.add((sample, cat, pred.strip()))
    return output

all_labels = set()
all_preds = set()
for sample in os.listdir('gens/one'):
    labels = process_labels(json.load(open(path.join('gens/one', sample, 'labels.json'))), int(sample))
    all_labels |= labels
    for cat in range(1, 10):
        preds = [x.strip() for x in open(path.join('gens/one', sample, 'cat_%d'%cat))]
        preds = process_preds(preds, int(sample), cat)
        all_preds |= preds

# method = 'approx-m'
method = 'em'
print(f1_score(all_labels, all_preds, method))

all_labels = set()
all_preds = set()
for sample in os.listdir('gens/zero'):
    labels = process_labels(json.load(open(path.join('gens/zero', sample, 'labels.json'))), int(sample))
    all_labels |= labels
    for cat in range(1, 10):
        preds = [x.strip() for x in open(path.join('gens/zero', sample, 'cat_%d'%cat))]
        preds = process_preds(preds, int(sample), cat)
        all_preds |= preds

print(f1_score(all_labels, all_preds, method))
