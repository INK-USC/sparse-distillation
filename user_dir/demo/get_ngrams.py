import argparse
import json
import os
import sys
import time
import itertools
import submitit
import pickle

from sklearn.feature_extraction.text import CountVectorizer
import multiprocessing


def process_one_split(all_args):
    args, corpus, idx = all_args[0], all_args[1], all_args[2]

    with open(args.dict, "rb") as fin:
        d = pickle.load(fin)
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 4), vocabulary=d)

    print("Thread {}: {} instances".format(idx, len(corpus)))

    start_time = time.time()
    X = vectorizer.transform(corpus)

    print("Thread {}: extraction finished after {} secs.".format(idx, time.time() - start_time))

    nrow, ncol = X.shape
    outputs = []

    for i in range(nrow):
        feature_list = []
        row_csr_matrix = X.getrow(i).sorted_indices()
        for feature_idx in row_csr_matrix.indices:
            feature_count = row_csr_matrix[0, feature_idx]
            # keep multiple appearance of the same feature in one instance
            for j in range(feature_count):
                feature_list.append(str(feature_idx)) 
        outputs.append(" ".join(feature_list)+"\n")

        if i % 10000 == 0:
            sys.stdout.write("Thread {}, processed {} instances , Time: {} sec\r".format(idx, i, time.time() - start_time))
            sys.stdout.flush()

    print("Thread {}: file writing finished after {} secs.".format(idx, time.time() - start_time))

    return outputs


def process_file(args, n_proc):
    print("Processing {} --> {}...".format(args.infile, args.outfile))

    with open(os.path.join(args.indir, args.infile)) as fin:
        all_sents = fin.readlines()

    if args.debug:
        all_sents = all_sents[:1000]

    print("File loading finished.")

    if args.n_proc > 1:
        executor = submitit.AutoExecutor(folder="submitit_log")
        executor.update_parameters(timeout_min=600, slurm_partition="debug", gpus_per_node=0, cpus_per_task=1)
        sents_per_proc = int(len(all_sents) / n_proc)
        sents_splits = [all_sents[i * sents_per_proc: (i+1) * sents_per_proc] for i in range(args.n_proc - 1)]
        sents_splits.append(all_sents[(args.n_proc-1) * sents_per_proc:])

        jobs = executor.map_array(process_one_split, [(args, sents_split, idx) for idx, sents_split in enumerate(sents_splits)])
        list_of_results = [job.result() for job in jobs]
        results = itertools.chain(*list_of_results)

    else:
        # job = executor.submit(process_one_split, (args, all_sents, 0))
        # results = job.result()
        results = process_one_split((args, all_sents, 0))

    with open(os.path.join(args.outdir, args.outfile), "w") as fout:
        fout.writelines(results)

def main(args):
    process_file(args, args.n_proc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='/mnt/nfs1/user/data/kd_raw2/raw_corpus_to_move/sst2')
    parser.add_argument('--outdir', default='/mnt/nfs1/user/data/kd_raw2/raw_corpus_to_move/sst2')
    parser.add_argument('--infile', default='dev.input0')
    parser.add_argument('--outfile', default='dev.ngram.input0')
    parser.add_argument('--dict', default='/mnt/nfs1/user/data/kd_raw/sst2/trigram_dict_ngram14_vocab1m.pkl')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--n_proc', type=int, default=1)
    args = parser.parse_args()
    main(args)