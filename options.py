import os
import argparse
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/')
    parser.add_argument('--ckpt')
    parser.add_argument('--aim_repo', default='.')
    parser.add_argument('--aim_exp', default='mimic-decisions-1215')
    parser.add_argument('--label_encoding', default='multiclass')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_losses', action='store_true')
    parser.add_argument('--task', default='token', choices=['seq', 'token'])
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--model', default='roberta-base',)
    parser.add_argument('--model_name', default='google/electra-base-discriminator',)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--grad_accumulation', default=2, type=int)
    parser.add_argument('--pheno_id', type=int)
    parser.add_argument('--unseen_pheno', type=int)
    parser.add_argument('--total_steps', type=int, default=5000)
    parser.add_argument('--train_log', type=int, default=500)
    parser.add_argument('--val_log', type=int, default=1000)
    parser.add_argument('--seed', default = '0')
    parser.add_argument('--num_phenos', type=int, default=10)
    parser.add_argument('--num_decs', type=int, default=9)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pos_weight', type=float, default=1.0)
    parser.add_argument('--truncate_train', action='store_true')
    parser.add_argument('--truncate_eval', action='store_true')
    parser.add_argument('--load_ckpt', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--resample', default='')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--use_crf', type=bool)


    args = parser.parse_args()

    curtime = datetime.now().strftime('%m%d_%H-%M-%S')
    #args.ckpt_dir = './checkpoints/%s-%s-%s'%(curtime, os.path.basename(args.model_name), args.model)
    args.ckpt_dir = '/checkpoints/%s-%s-%s'%(curtime, os.path.basename(args.model_name), args.model)
    args.seed = [int(x) for x in args.seed.split(',')]

    if args.task == 'seq' and args.pheno_id is not None:
        args.num_labels = 1
    elif args.task == 'seq':
        args.num_labels = args.num_phenos
    elif args.task == 'token':
        args.num_labels = args.num_decs 
        if args.label_encoding == 'multiclass':
            args.num_labels = args.num_labels * 2 + 1 # 9*2+1=19
        elif args.label_encoding == 'bo':
            args.num_labels *= 2
        elif args.label_encoding == 'boe':
            args.num_labels *= 3

    return args
