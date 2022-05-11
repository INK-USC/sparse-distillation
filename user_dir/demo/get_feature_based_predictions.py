from user_dir.models.feature_based_linear_model import FeatureBasedLinearModel
from user_dir.data.feature_dropout_dataset import FeatureDropoutDataset
from fairseq.data.data_utils import collate_tokens
from tqdm import tqdm
from fairseq.data.shorten_dataset import maybe_shorten_dataset


import argparse
import torch
import numpy as np
import pickle
import time
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import os

from fairseq.data import (
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    StripTokenDataset,
    data_utils,
)

logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

MAX_LEN = 2048

def load_model(args):
    model = FeatureBasedLinearModel.from_pretrained(
        args.model_path,
        checkpoint_file=args.model_name,
        data_name_or_path=args.data_path
    )
    if args.fp16:
        model.half()
    if not args.cpu:
        model.cuda()
    model.eval()  # disable dropout
    return model

def truncate(item):
    return item[:MAX_LEN]

def predict_batch(batch, model):
    tokens, lens = batch
    with torch.no_grad():
        pred = model.predict('imdb_head', tokens, lens, return_logits=True)
    return pred

def get_batches(src_tokens, src_lengths, dictionary):

    num_batches = len(src_tokens) // args.batch_size

    batches = []

    for i in range(num_batches):
        st = i * args.batch_size
        ed = (i+1) * args.batch_size
        tokens = collate_tokens([src_tokens[j] for j in range(st, ed)], pad_idx=dictionary.pad())
        lengths = torch.LongTensor([src_lengths[j] for j in range(st,ed)])
        batches.append((tokens, lengths))

    if num_batches * args.batch_size < len(src_tokens):
        st = num_batches * args.batch_size
        ed = len(src_tokens)
        tokens = collate_tokens([src_tokens[j] for j in range(st, ed)], pad_idx=1)
        lengths = torch.LongTensor([src_lengths[j] for j in range(st,ed)])
        batches.append((tokens, lengths))
    
    # print(len(batches))
    return batches


def get_labels(args):
    with open(args.label_file, "r") as fin:
        labels = fin.readlines()
    labels = [int(item.strip()) for item in labels]
    return np.array(labels)

def load_fairseq_dataset(args):

    dictionary = Dictionary.load(os.path.join(args.data_path, "input0", "dict.txt"))
    dictionary.add_symbol("<mask>")

    # print("[input] dictionary: {} types".format(len(dictionary)))
    # print(dictionary.pad())

    src_tokens = data_utils.load_indexed_dataset(
        os.path.join(args.data_path, "input0", "valid"),
        dictionary,
        None,
        combine=False,
    )
    src_tokens = StripTokenDataset(src_tokens, id_to_strip=dictionary.eos())
    
    src_tokens = FeatureDropoutDataset(
        src_tokens, 
        dropout=0.0, # don't use feature dropout for validation
        truncation_length=MAX_LEN, 
        seed=0
    )

    src_lengths = NumelDataset(src_tokens, reduce=False)

    label_dict = Dictionary.load(os.path.join(args.data_path, "label", "dict.txt"))
    label_dict.add_symbol("<mask>")
    labels = data_utils.load_indexed_dataset(
        os.path.join(args.data_path, "label", "valid"),
        label_dict,
        None,
        combine=False
    )

    labels = OffsetTokensDataset(
                StripTokenDataset(
                    labels,
                    id_to_strip=label_dict.eos(),
                ),
                offset=-label_dict.nspecial,
            )

    lst_labels = np.array([labels[i].item() for i in range(len(labels))])
    # print(lst_labels[:10])
    return src_tokens, src_lengths, lst_labels, dictionary, label_dict

def post_process(preds, label_dict):
    for i in range(len(preds)):
        preds[i] = int(label_dict.string([preds[i] + label_dict.nspecial]))
    return preds

def main(args):

    src_tokens, src_lengths, labels, dictionary, label_dict = load_fairseq_dataset(args)
    batches = get_batches(src_tokens, src_lengths, dictionary)

    print(batches[0])

    # model = load_model(args)

    # start = time.time()

    # preds = []
    # for batch in tqdm(batches):
    #     output = predict_batch(batch, model)
    #     preds += output
    # end = time.time()

    # print(end-start)

    # preds = torch.stack(preds).cpu().numpy()

    # preds = preds.argmax(axis=1)
    # preds = post_process(preds, label_dict)

    # labels = get_labels(args)

    # acc = accuracy_score(labels, preds)
    
    # print("acc:{}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--label_file', default='/mnt/nfs1/user/data/raw/imdb/aclImdb/dev.label')
    parser.add_argument('--model_path', default='/mnt/nfs1/user/checkpoints/0624-imdb-1mdict-bsz2048-param1b-1msteps')
    parser.add_argument('--model_name', default='checkpoint_best.pt')
    parser.add_argument('--data_path', default='/mnt/nfs1/user/data/kd/imdb/imdb-n-amazon-all-1mdict-bin')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    print(args)
    main(args)