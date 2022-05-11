import argparse
import json
import os

from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm, trange
import pickle

def process_train(args, vectorizer):
    print("Processing train data...")

    with open(os.path.join(args.datadir, "train.input0")) as fin:
        corpus = fin.readlines()

    print("Loaded {} lines".format(len(corpus)))

    if args.debug:
        corpus = corpus[:50]

    X = vectorizer.fit(corpus)
    print("vocab size: {}".format(len(vectorizer.vocabulary_)))

def save_dict(args, vectorizer):
    with open(os.path.join(args.datadir, "trigram_dict_ngram14_vocab1m.pkl"), "wb") as fout:
        pickle.dump(vectorizer.vocabulary_, fout)

def main(args):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 4), max_features=1000000)
    process_train(args, vectorizer)
    save_dict(args, vectorizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='../data/imdb_n_amazon_raw')
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    main(args)