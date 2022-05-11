from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens
from tqdm import tqdm

import torch.multiprocessing as mp
from multiprocessing import Process, Manager

import argparse
import torch
import numpy as np

import sys
import time

import submitit

MAX_LEN = 512

def load_model(args, idx):
    roberta = RobertaModel.from_pretrained(
        args.model_path,
        checkpoint_file=args.model_name,
        data_name_or_path=args.data_path
    )
    roberta.cuda(device=idx)
    roberta.half()
    roberta.eval()  # disable dropout
    return roberta

def random_dropout(item):
    """Dropout features if input is longer than MAX_LEN"""
    perm = np.random.permutation(len(item))
    idx = perm[:MAX_LEN]
    item = item[idx]
    return item

def truncate(item):
    return item[:MAX_LEN]

def predict_batch(sents, model):
    batch = collate_tokens(
        [truncate(model.encode(sent)) for sent in sents], pad_idx=1
    )
    with torch.no_grad():
        pred = model.predict('sentence_classification_head', batch, return_logits=True)

    return pred

def get_batches(args, sents):
    num_batches = len(sents) // args.batch_size

    batches = []
    for i in range(num_batches):
        batches.append(sents[i * args.batch_size: (i+1) * args.batch_size])
    if num_batches * args.batch_size < len(sents):
        batches.append(sents[num_batches * args.batch_size: ])
    
    return batches

def process_one_split(args, idx, data):
    torch.manual_seed(0)
    np.random.seed(0)

    print("Starting Thread {}".format(idx))
    roberta = load_model(args, 0)
    batches = get_batches(args, data)
    print("Thread {}: Model loaded.".format(idx))

    start_time = time.time()

    preds = []
    for idx_b, batch in enumerate(batches):
        output = predict_batch(batch, roberta)
        preds += output
        
        if idx_b % 200 == 0:
            print("Thread {}, processed {}/{} batches , Time: {} sec\r".format(idx, idx_b, len(batches), time.time() - start_time))
            # sys.stdout.flush()

    preds = torch.stack(preds).cpu().numpy()
    return preds

def main(args):
    with open(args.in_file) as fin:
        all_sents = fin.readlines()

    if args.debug:
        all_sents = all_sents[:1000]

    print("loaded {} lines from {}".format(len(all_sents),args.in_file))

    executor = submitit.AutoExecutor(folder="submitit_logs")
    executor.update_parameters(timeout_min=720, slurm_partition="a100", gpus_per_node=1, slurm_exclude="a100-st-p4d24xlarge-37,a100-st-p4d24xlarge-86")

    if args.n_proc > 1:
        sents_per_proc = int(len(all_sents) / args.n_proc)
        sents_split = [all_sents[i * sents_per_proc: (i+1) * sents_per_proc] for i in range(args.n_proc - 1)]
        sents_split.append(all_sents[(args.n_proc-1) * sents_per_proc:])

        lst_args, lst_idx = [], []
        for i in range(args.n_proc):
            lst_args.append(args)
            lst_idx.append(i)

        jobs = executor.map_array(process_one_split, lst_args, lst_idx, sents_split)

        preds = np.concatenate([job.result() for job in jobs])
        # print(preds)

    else:
        # preds = process_one_split(args, 0, all_sents)
        job = executor.submit(process_one_split, args, 0, all_sents)

        preds = job.result()
        # print(preds)

    preds = preds.astype(np.float32)
    with open(args.out_file, "wb") as fout:
        np.save(fout, preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_file', default='../data/s140/s140_all_amazon_reviews_raw/train.input0')
    parser.add_argument('--out_file', default='../data/s140/s140_all_amazon_reviews_raw/train.roberta.distill.parallel.npy')
    parser.add_argument('--model_path', default='../../checkpoints/roberta-ft-s140')
    parser.add_argument('--model_name', default='checkpoint_best.pt')
    parser.add_argument('--data_path', default='/data/user/yeqy/src/fairseq/binarized/S140-bin/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--debug', action="store_true")
    
    args = parser.parse_args()

    print(args)
    main(args)