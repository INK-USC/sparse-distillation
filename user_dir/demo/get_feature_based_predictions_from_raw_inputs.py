from user_dir.models.feature_based_linear_model import FeatureBasedLinearModel

from fairseq.data import Dictionary
from fairseq.data.data_utils import collate_tokens

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from tqdm import tqdm

import argparse
import torch
import numpy as np
import pickle
import time
import logging
import os

logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

MAX_LEN = 2048

def load_model(args):
    model = FeatureBasedLinearModel.from_pretrained(
        args.model_path,
        checkpoint_file=args.model_name,
        data_name_or_path=args.model_path,
    )
    model.cuda()
    model.eval()  # disable dropout
    return model

def encode(model, sent):
    return model.model.encoder.dictionary.encode_line(sent, add_if_not_exist=False).long()

def truncate(item):
    return item[:-1][:MAX_LEN] #[:-1] is to remove the <end of sentence> token at the end

def predict_batch(sents, model):
    tokenized_sents = [truncate(encode(model, sent)) for sent in sents]
    lens = torch.LongTensor([len(tokenized_sent) for tokenized_sent in tokenized_sents])

    batch = collate_tokens(
        tokenized_sents, pad_idx=1
    )
    
    with torch.no_grad():
        pred = model.predict('imdb_head', batch, lens)
    return pred

def extract_features(args, feature_extractor):

    start_time = time.time()

    with open(args.in_file) as fin:
        sents = fin.readlines()

    X = feature_extractor.transform(sents)

    print("extraction finished after {} secs.".format(time.time() - start_time))

    nrow, ncol = X.shape
    outputs = []

    # turn sparse matrix into a list of features (in a brute-force way)
    for i in range(nrow):
        feature_list = []
        row_csr_matrix = X.getrow(i).sorted_indices()
        for feature_idx in row_csr_matrix.indices:
            feature_count = row_csr_matrix[0, feature_idx]
            # keep multiple appearance of the same feature in one instance
            for j in range(feature_count):
                feature_list.append(str(feature_idx)) 
        outputs.append(" ".join(feature_list))

    print("post processing finished after {} secs.".format(time.time() - start_time))

    return outputs

def get_batches(args, sents):
    num_batches = len(sents) // args.batch_size

    batches = []
    for i in range(num_batches):
        
        batches.append(sents[i * args.batch_size: (i+1) * args.batch_size])
    if num_batches * args.batch_size < len(sents):
        batches.append(sents[num_batches * args.batch_size: ])

    print("{} batches in total with batch size of {}".format(len(batches), args.batch_size))
    return batches


def get_vectorizer(args):
    with open(args.feature_dict_path, "rb") as fin:
        d = pickle.load(fin)
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 4), vocabulary=d)
    return vectorizer

def get_labels(args):
    with open(args.label_file, "r") as fin:
        labels = fin.readlines()
    labels = [int(item.strip()) for item in labels]
    return np.array(labels)

def post_process(preds, label_dict):
    for i in range(len(preds)):
        preds[i] = int(label_dict.string([preds[i] + label_dict.nspecial]))
    return preds
    
def main(args):

    # load n-gram vocabulary
    vectorizer = get_vectorizer(args)

    # load raw text from args.in_file, then extract n-grams from raw text
    sents = extract_features(args, vectorizer)

    # create batches
    batches = get_batches(args, sents)

    # load model
    print("loading model")
    model = load_model(args)
    print("loading model finished")

    # make predictions
    preds = []
    for batch in tqdm(batches):
        output = predict_batch(batch, model)
        preds += output

    # post-process predictions
    preds = torch.stack(preds).cpu().numpy()
    preds = preds.argmax(axis=1)
    label_dict = Dictionary.load(os.path.join(args.model_path, "label", "dict.txt"))
    preds = post_process(preds, label_dict)

    # get true labels
    labels = get_labels(args)
    
    # compute accruacy
    acc = accuracy_score(labels, preds)
    print("acc:{}".format(acc))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_file', 
        help="input file. each line is an input example.",
        default='/mnt/nfs1/user/data/raw/imdb/aclImdb/dev.input0')
    parser.add_argument('--label_file', 
        help="label file. each line is a number representing the label. should have the same number of lines with --in_file",
        default='/mnt/nfs1/user/data/raw/imdb/aclImdb/dev.label')
    parser.add_argument('--model_path', 
        help="directory to the model",
        default='/mnt/nfs1/user/checkpoints/0624-imdb-1mdict-bsz2048-param1b-1msteps')
    parser.add_argument('--model_name', 
        default='checkpoint_best.pt')
    parser.add_argument('--feature_dict_path', 
        help="mapping of n-grams and the indices",
        default='/mnt/nfs1/user/data/kd_raw/imdb/trigram_dict_ngram14_vocab1m.pkl')
    parser.add_argument('--batch_size', default=8)
    args = parser.parse_args()

    print(args)
    main(args)