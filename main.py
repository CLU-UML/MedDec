import os
import aim
import json
import torch
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from tqdm import tqdm
from data import load_data
from model import load_model
from options import get_args

mean = lambda l: sum(l)/len(l) if len(l) > 0 else .0

args = get_args()

# device = 'cuda:%s'%args.gpu
device = 'cuda' if torch.cuda.is_available() else 'mps'
all_losses = {'train': [], 'val': [], 'test': []}


def indicators_to_spans(labels, idx = None) -> set:
    """    
    Convert label indicators to spans.

    Args:
        labels (list or numpy.ndarray): The label indicators. If `args.label_encoding` is 'multiclass', 
            it should be a list of integers. Otherwise, it should be a 2D numpy array.
        idx (int, optional): An optional index to include in the span. Defaults to None.

    Returns:
        spans: A set of spans, where each span is a tuple (idx, category, start, end).

    The function processes the label indicators based on the label encoding specified in `args.label_encoding` 
    and converts them into spans. The spans are represented as tuples containing the index, category, start, 
    and end positions. The function supports different label encodings such as 'multiclass', 'bo', and 'boe'.
    """

    def add_span(idx, c, start, end):
        """
        Add single span to spans, where span is a set: (idx, c, start, end)
        """
        span = (idx, c, start, end)
        spans.add(span)

    spans = set()
    if args.label_encoding == 'multiclass':
        num_tokens = len(labels)
        num_classes = args.num_labels // 2
        start = None
        cat = -1
        for t in range(num_tokens):
            prev_tag = labels[t-1] if t > 0 else args.num_labels -1 
            cur_tag = labels[t]

            if start is not None and cur_tag == cat + 1:
                continue
            elif start is not None:
                add_span(idx, cat // 2, start, t - 1) # end = t-1
                start = None

            if start is None and (cur_tag in [2*x for x in range(num_classes)]
                              or (prev_tag == (args.num_labels - 1)
                                  and cur_tag != (args.num_labels - 1))):
                start = t
                cat = int(cur_tag) // 2 * 2
    else: # not multiclass
        num_tokens, num_classes = labels.shape
        if args.label_encoding == 'bo':
            num_classes //= 2
        elif args.label_encoding == 'boe':
            num_classes //= 3

        for c in range(num_classes):
            start = None
            for t in range(num_tokens):
                if args.label_encoding == 'bo':
                    if start and (labels[t, 2 * c] == 1 or labels[t, 2 * c + 1] == 0):
                        add_span(idx, c, start, t - 1)
                        start = None
                    elif start and labels[t, 2 * c + 1] == 1:
                        continue
                    if labels[t, 2 * c] == 1:
                        start = t
                elif args.label_encoding == 'boe':
                    if not start and labels[t, 3 * c] == 1:
                        start = t
                    elif start and labels[t, 3 * c + 2] == 1:
                        add_span(idx, c, start, t - 1)
                        start = None
                else:
                    if start and labels[t,c] == 1:
                        continue
                    elif start and labels[t,c] == 0:
                        add_span(idx, c, start, t - 1)
                        start = None
                    elif labels[t,c] == 1 and t == (num_tokens - 1):
                        span = (idx, c, -1, -1)
                        spans.add(span)
                    elif labels[t,c] == 1:
                        start = t
    return spans


def id_to_label(labels) -> list:
    """
    Converts a list of label IDs to their corresponding label strings.
    Args:
        labels (list of int): A list of label IDs.
    Returns:
        list of str: A list of label strings where:
            - 'O' represents the label ID for 'Other'.
            - 'B-x' represents the beginning of a chunk with label ID x.
            - 'I-x' represents the inside of a chunk with label ID x.
    """
    new_labels = []
    for l in labels:
        if l == (args.num_labels - 1):
            new_l = 'O'
        elif l % 2 == 0:
            new_l = 'B-%d'% (l // 2)
        else:
            new_l = 'I-%d'% (l // 2)
        new_labels.append(new_l)
    return new_labels

def f1_score(ys, preds):
    """
    Compute F1-Score of the predictions
    """
    tp = len(preds & ys)
    fn = len(ys) - tp
    fp = len(preds) - tp
    f1 = (2 * tp / (2 * tp + fp + fn)) * 100 if tp + fp + fn > 0 else 0
    return f1

def recall_score(ys, preds):
    """
    Compute Recall of the predictions
    """
    tp = len(preds & ys)
    fn = len(ys) - tp
    recall = tp / (tp + fn)
    return recall

def calc_metrics_spans(ys, preds, span_ys = None):
    """
    Calculate F1 score and span metrics for predictions.

    Args:
        ys (list): List of ground truth labels.
        preds (list): List of predicted labels.
        span_ys (list, optional): List of ground truth spans. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - f1 (float): The overall F1 score.
            - all_preds (set): A set of all predicted spans.
            - all_ys (set): A set of all ground truth spans.
            - perclass (dict): A dictionary with F1 scores per class.
    """
    all_preds = []
    all_ys = []
    for i, (y, pred) in enumerate(zip(ys, preds)):
        pred_spans = indicators_to_spans(pred, idx = i)
        all_preds.append(pred_spans)
        if span_ys is None:
            y = y.squeeze()
            y_spans = indicators_to_spans(y, idx = i)
            all_ys.append(y_spans)

    all_preds = set().union(*all_preds)
    if span_ys is None:
        all_ys = set().union(*all_ys)
    else:
        all_ys = set(span_ys)
    f1 = f1_score(all_ys, all_preds)

    perclass = {}
    for c in range(args.num_decs):
        sub_ys = {x for x in all_ys if x[1] == c}
        sub_preds = {x for x in all_preds if x[1] == c}
        perclass[c] = f1_score(sub_ys, sub_preds)

    return f1, all_preds, all_ys, perclass

def save_losses(model, crit, train_dataloader, val_dataloader, test_dataloader):
    """
    Save losses for train, val, and test sets
    """
    train_losses = evaluate(model, train_dataloader, crit, return_losses = True)
    all_losses['train'].append(train_losses)
    val_losses = evaluate(model, val_dataloader, crit, return_losses = True)
    all_losses['val'].append(val_losses)
    test_losses = evaluate(model, test_dataloader, crit, return_losses = True)
    all_losses['test'].append(test_losses)

def evaluate(model, dataloader, crit, return_losses = False, return_preds = False):
    """
    Evaluates the given model on the provided dataloader using the specified criterion.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader providing the evaluation data.
        crit (torch.nn.Module): The criterion (loss function) used for evaluation.
        return_losses (bool, optional): If True, returns the individual losses for each batch. Defaults to False.
        return_preds (bool, optional): If True, returns the predictions and corresponding spans. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - metrics_out (dict): A dictionary containing evaluation metrics such as 'f1' and 'acc'.
            - pheno_results (dict or None): A dictionary containing phenotype-specific F1 scores if applicable, 
                otherwise None.
            - loss (torch.Tensor): The mean loss over the evaluation dataset.
            - perclass (dict): A dictionary containing per-class evaluation metrics.
            - (optional) losses (list): A list of individual losses for each batch if return_losses is True.
            - (optional) span_preds (list): A list of predicted spans if return_preds is True.
            - (optional) span_ys (list): A list of ground truth spans if return_preds is True.
    """
    model.eval()
    outs, ys = [], []
    lens = []
    token_masks = []
    for batch in tqdm(dataloader, desc='Evaluation'):
        x = batch['input_ids']
        y = batch['labels']
        mask = batch['mask']
        if args.task == 'seq':
            ids = batch['ids']

        with torch.no_grad():
            logits = model.generate(x, mask)

        outs.append(logits)
        lens.extend([x.shape[0] for x in logits])
        ys.append(y)

        if 'token_mask' in batch:
            token_masks.append(batch['token_mask'])

    if args.label_encoding == 'multiclass':
        outs_stack = torch.cat([x.view(-1, args.num_labels) for x in outs], 0)
        ys_stack = torch.cat([x.view(-1) for x in ys], 0).to(device)
        preds = [x.squeeze() for x in outs]

        if args.use_crf:
            padded_outs = torch.nn.utils.rnn.pad_sequence(preds, batch_first=True)
            outs_mask = ~(padded_outs[:,:,0] == 0)
            preds = crit.decode(padded_outs, mask=outs_mask)
            preds_stack = torch.tensor([x for pred in preds for x in pred]).to(device)
            padded_ys = torch.nn.utils.rnn.pad_sequence([x.squeeze() for x in ys], batch_first=True)
            loss = -1 * crit(padded_outs, padded_ys, mask=outs_mask, reduction='mean')

        else:
            preds =  [x.argmax(-1) for x in preds]
            preds_stack = outs_stack.argmax(-1)
            loss = crit(outs_stack, ys_stack)
    else:
        outs_stack = torch.cat(outs, 1)
        ys_stack = torch.cat(ys, 1).to(device)
        loss = crit(outs_stack, ys_stack)

    losses = []
    offset = 0
    if return_losses:
        for ln in lens:
            sub_losses = loss[offset:offset+ln]
            offset += ln
            losses.append(sub_losses.mean().item())
        return losses


    loss = loss.mean()

    y = torch.cat(ys, 1).squeeze()

    if len(token_masks) > 0:
        token_masks = torch.cat(token_masks, 1).squeeze().to(torch.int32).to(device)
        acc = ((ys_stack == preds_stack).float() * token_masks).sum() / token_masks.sum() * 100
    else:
        acc = (ys_stack == preds_stack).float().mean() * 100

    if 'all_spans' in dataloader.dataset.data[0]:
        all_spans = [x['all_spans'] for x in dataloader.dataset.data]
        span_ys = [(i, s['label'], s['token_start'], s['token_end']) for i, spans in enumerate(all_spans) for s in spans[0]]
    else:
        span_ys = None
    print('Calculating metrics...')
    f1, span_preds, span_ys, perclass = calc_metrics_spans(ys, preds, span_ys)
    if return_preds:
        return span_preds, span_ys
    metrics_out = {}
    metrics_out['f1'] = f1
    metrics_out['acc'] = acc
    model.train()

    # genders = dataloader.dataset.stats['gender']
    # for g in set(genders):
    #     ids = [i for i,x in enumerate(genders) if x==g]
    #     sub_ys = torch.cat([x for i,x in enumerate(ys) if i in ids], 1).squeeze().cpu()
    #     sub_preds = torch.cat([x for i,x in enumerate(preds) if i in ids]).cpu()
    #     sub_acc = (sub_ys == sub_preds).float().mean() * 100
    #     print(g, sub_acc)

    # ethnicities = dataloader.dataset.stats['ethnicity']
    # for e in set(ethnicities):
    #     ids = [i for i,x in enumerate(ethnicities) if x==e]
    #     sub_ys = torch.cat([x for i,x in enumerate(ys) if i in ids], 1).squeeze().cpu()
    #     sub_preds = torch.cat([x for i,x in enumerate(preds) if i in ids]).cpu()
    #     sub_acc = (sub_ys == sub_preds).float().mean() * 100
    #     print(e, sub_acc)

    # langs = dataloader.dataset.stats['language']
    # for l in set(langs):
    #     ids = [i for i,x in enumerate(langs) if x==l]
    #     sub_ys = torch.cat([x for i,x in enumerate(ys) if i in ids], 1).squeeze().cpu()
    #     sub_preds = torch.cat([x for i,x in enumerate(preds) if i in ids]).cpu()
    #     sub_acc = (sub_ys == sub_preds).float().mean() * 100
    #     print(l, sub_acc)

    if args.task == 'token':
        print('Calculating pheno results...')
        pheno_results = {}
        for pheno, ids in dataloader.dataset.pheno_ids.items():
            sub_ys = [x for i,x in enumerate(ys) if i in ids]
            sub_preds = [x for i,x in enumerate(preds) if i in ids]
            f1, span_preds, span_ys, _ = calc_metrics_spans(sub_ys, sub_preds)
            pheno_results[pheno] = f1
    else:
        pheno_results = None
    return metrics_out, pheno_results, loss, perclass

def process(sample, model, tokenizer, out_dir):
    """
    Processes a given sample using a specified model and tokenizer, and saves the output to a JSON file.

    Args:
        sample (dict): A dictionary containing the sample data with keys 'HADM_ID', 'SUBJECT_ID', 'ROW_ID', and 'TEXT'.
        model (torch.nn.Module): The model used for generating predictions.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for encoding the sample text.
        out_dir (str): The directory where the output JSON file will be saved.

    Returns:
        None
    """
    hadm = sample['HADM_ID']
    fn = f"{sample['SUBJECT_ID']}_{int(hadm) if not pd.isnull(hadm) else 'NaN'}_{sample['ROW_ID']}.json"
    out_file = out_dir + fn
    if not os.path.exists(out_file):
        encoding = tokenizer.encode_plus(sample['TEXT'])
        x = torch.tensor(encoding['input_ids']).unsqueeze(0).to(device)
        mask = torch.tensor(encoding['attention_mask']).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model.generate(x, mask)

        if args.label_encoding == 'multiclass':
            pred = out.argmax(-1)
        else:
            pred = torch.where(out > 0, 1, 0)
        pred = pred.squeeze()
        spans = indicators_to_spans(pred)
        all_spans = []
        for _, cat, start, end in spans:
            span_dict = {}
            span_dict['decision'] = sample['TEXT'][start:end]
            span_dict['category'] = 'Category %d'%(cat+1)
            span_dict['start_offset'] = start
            span_dict['end_offset'] = end
            all_spans.append(span_dict)
        with open(out_file, 'w') as f:
            json.dump(all_spans, f)

def predict_mimic(model, data, tokenizer):
    """
    Predicts medical decisions using the given model and data.

    Args:
        model (torch.nn.Module): The trained model to use for predictions.
        data (iterable): The dataset to predict on.
        tokenizer (Tokenizer): The tokenizer to preprocess the data.

    Returns:
        None

    This function sets the model to evaluation mode and processes the data in parallel using multiprocessing.
    The results are saved in the specified output directory.
    """

    model.eval()
    outs = []
    out_dir = './all_mimic_decisions/'
    kwargs = {'model': model, 'tokenizer': tokenizer, 'out_dir': out_dir}
    import multiprocessing as mp
    from functools import partial
    mp.set_start_method('spawn', force=True)
    process_ = partial(process, **kwargs)
    pool = mp.Pool(3)
    for _ in tqdm(pool.imap_unordered(process_, data, chunksize=20000), total=len(data)):
        pass
    pool.close()
    # for sample in tqdm(data):
    #     process(sample)

def train(args, model, crit, optimizer, lr_scheduler, train_dataloader, val_dataloader, 
          verbose=True, train_ns=None, test_dataloader=None):
    """
    Train the model with the given parameters.
    Args:
        args (Namespace): Arguments containing training configurations.
        model (torch.nn.Module): The model to be trained.
        crit (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        train_dataloader (DataLoader): DataLoader for the training data.
        val_dataloader (DataLoader): DataLoader for the validation data.
        verbose (bool, optional): If True, prints training progress. Defaults to True.
        train_ns (optional): Additional training namespace. Defaults to None.
        test_dataloader (DataLoader, optional): DataLoader for the test data. Defaults to None.
    Returns:
        tuple: Best F1 score, best accuracy, and the step at which the best F1 score was achieved.
    """
    writer = aim.Run(experiment=args.aim_exp, repo=args.aim_repo, 
            system_tracking_interval=0) if not args.debug else None
    if writer is not None:
        writer['hparams'] = args.__dict__

    step = 0
    best_f1 = -1
    best_acc = 0
    best_step = 0
    best_pheno = None
    best_perclass = None
    train_iter = iter(train_dataloader)
    losses = []
    while step < args.total_steps:
        batch = next(train_iter, None)
        if batch is None:
            train_iter = iter(train_dataloader)
            continue
        x = batch['input_ids']
        y = batch['labels']
        mask = batch['mask']

        y = y.to(device)
        if args.task == 'seq':
            out, _ = model.phenos(x, mask)
            logits = out[1]
        elif args.task == 'token':
            out, logits = model.decisions(x, mask)

        if args.label_encoding == 'multiclass':
            if args.use_crf:
                loss = -1 * crit(logits, y, reduction='mean')
            else:
                loss = crit(logits.view(-1, args.num_labels), y.view(-1)).mean()
        else:
            loss = crit(logits, y).mean()
        total_loss = loss


        losses.append(loss.item())
        total_loss /= args.grad_accumulation
        total_loss.backward(retain_graph=True)
        
        if (step+1) % args.grad_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        if step % (args.train_log*args.grad_accumulation) == 0:
            avg_loss = np.mean(losses)
            if verbose:
                print('step %d - training loss: %.3f'%(step, avg_loss))
            if writer is not None:
                writer.track(avg_loss, name='bce_loss', context={'split': 'train'}, step = step)
            losses = []

        if len(val_dataloader) > 0 and step % (args.val_log*args.grad_accumulation) == 0:
            if args.save_losses:
                save_losses(model, crit, train_ns, val_dataloader, test_dataloader)
            metrics_out, pheno_results, loss, perclass = evaluate(model, val_dataloader, crit)
            f1, acc = metrics_out['f1'], metrics_out['acc']
            if verbose:
                print('[step: {:5d}] f1: {:.1f}, acc: {:.1f}, loss: {:.3f}'
                        .format(step, f1, acc, loss))
            if writer is not None:
                writer.track(loss, name='bce_loss', context={'split': 'val'}, step = step)
                writer.track(f1, name='f1', step = step)
                # writer.track(prec, name='precision', step = step)
                # writer.track(rec, name='recall', step = step)
            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc
                best_step = step
                best_pheno = pheno_results
                # best_perclass = metrics_out[4:6]
                if not args.debug:
                    torch.save(model.state_dict(), args.ckpt_dir)
        step += 1
    if writer is not None:
        writer.track(best_f1, name = 'best_f1')
        writer.track(best_step, name = 'best_step')
        if best_pheno is not None:
            for pheno, f1 in best_pheno.items():
                writer.track(f1, name='best_f1', context={'group': pheno})
        if args.task == 'token':
            f1s = best_perclass
            for i in range(len(f1s)):
                writer.track(f1s[i], name='best_f1', context={'decision': i})
    return best_f1, best_acc, best_step

def main(args):
    """
    Main function to train and evaluate a model based on the provided arguments.

    Args:
        args (Namespace): A namespace object containing various arguments and configurations 
                          for training and evaluation. Expected attributes include:
                          - seed (list of int): List of random seeds for reproducibility.
                          - eval_only (bool): Flag to indicate if only evaluation should be performed.
                          - verbose (bool): Flag to control verbosity of training output.
                          - save_losses (bool): Flag to indicate if losses should be saved.

    Returns:
        float: The mean F1 score across different seeds.
    """
    f1s = []
    for seed in args.seed:
        print('-----------------------------------')
        print('Seed:', seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        args.seed = seed
        train_dataloader, val_dataloader, test_dataloader, train_ns = load_data(args)
        print('Data loaded')
        model, crit, optimizer, lr_scheduler = load_model(args, device)
        print('Model loaded')

        if not args.eval_only:
            f1, acc, step = train(args, model, crit, 
                    optimizer, lr_scheduler, train_dataloader,
                    val_dataloader, args.verbose, train_ns, test_dataloader)
            f1s.append(f1)
            print('seed: %d, F1: %.1f, Acc: %.1f'%(seed, f1, acc))
            # Test
            metrics_out, pheno_results, loss, perclass = evaluate(model, test_dataloader, crit)
            f1, acc = metrics_out['f1'], metrics_out['acc']
            print('[Test] f1: {:.1f}, acc: {:.1f}, loss: {:.3f}'
                    .format(f1, acc, loss))
            # print(pheno_results)
            print(perclass)
        else:
            model.eval()
            # Train
            # metrics_out, pheno_results, loss = evaluate(model, train_ns, crit)
            # f1, acc = metrics_out['f1'], metrics_out['acc']
            # print('[Train] f1: {:.1f}, acc: {:.1f}, loss: {:.3f}'
            #         .format(f1, acc, loss))

            # Val
            # metrics_out, pheno_results, loss, perclass = evaluate(model, val_dataloader, crit)
            # f1, acc = metrics_out['f1'], metrics_out['acc']
            # print('[Val] f1: {:.1f}, acc: {:.1f}, loss: {:.3f}'
            #         .format(f1, acc, loss))

            # Test
            metrics_out, pheno_results, loss, perclass = evaluate(model, test_dataloader, crit)
            f1, acc = metrics_out['f1'], metrics_out['acc']
            print('[Test] f1: {:.1f}, acc: {:.1f}, loss: {:.3f}'
                    .format(f1, acc, loss))
            # print(pheno_results)
            print(perclass)

            # predict_mimic(model, data, tokenizer)
        if args.save_losses:
            np.savez('losses_%d.npz'%seed, train=all_losses['train'], val=all_losses['val'], test=all_losses['test'])
    return np.mean(f1s)

if __name__ == '__main__':
    main(args)

# python main.py --data_dir "data_dir/" --label_encoding multiclass --model_name google/electra-base-discriminator --total_steps 5000 --lr 4e-5 --eval_only