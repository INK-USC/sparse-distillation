import json
import os
import shutil
import glob
import gzip

DATA_PATH="/data/home/user/data/amazon_review_18/*.json.gz"
OUT_DIR="/data/home/user/src/fairseq/data/imdb_n_amazon_all_categories_raw"
OUT_PATH="/data/home/user/src/fairseq/data/imdb_n_amazon_all_categories_raw/train.input0"
# a placeholder label file so that binarization won't go wrong;
# distilation doesn't need labels
OUT_PATH_LABEL="/data/home/user/src/fairseq/data/imdb_n_amazon_all_categories_raw/train.label"
IMDB_DIR="/data/home/user/src/fairseq/data/aclImdb/"

def main():
    kept = 0
    discarded = 0

    os.makedirs(OUT_DIR, exist_ok=True)

    # all categories
    filenames = glob.glob(DATA_PATH)

    # one review per line.
    for filename in filenames:
        print("Processing {}".format(filename))
        with gzip.open(filename, "rb") as fin, open(OUT_PATH, "a") as fout1, open(OUT_PATH_LABEL, "a") as fout2:
            for line in fin:
                d = json.loads(line)
                if "reviewText" in d:
                    fout1.write(d["reviewText"].replace("\n", " ").replace("\t", " ").replace("\r", " ").strip() + "\n")
                    fout2.write("0\n")
                    kept += 1
                else:
                    discarded += 1
        
        print("Finish Processing {}. Kept: {}, Discarded: {}".format(filename, kept, discarded))

    # add original imdb data (unsupervised part)
    with open(os.path.join(IMDB_DIR, "unsup.input0")) as fin1, \
        open(OUT_PATH, "a") as fout1, open(OUT_PATH_LABEL, "a") as fout2:
        for line1 in fin1:
            fout1.write(line1)
            fout2.write("0\n")

    # add original imdb data (train)
    with open(os.path.join(IMDB_DIR, "train.input0")) as fin1, open(os.path.join(IMDB_DIR, "train.label")) as fin2, \
        open(OUT_PATH, "a") as fout1, open(OUT_PATH_LABEL, "a") as fout2:
        for line1, line2 in zip(fin1, fin2):
            fout1.write(line1)
            fout2.write(line2)

    shutil.copyfile(os.path.join(IMDB_DIR, "dev.input0"), os.path.join(OUT_DIR, "dev.input0"))
    shutil.copyfile(os.path.join(IMDB_DIR, "dev.label"), os.path.join(OUT_DIR, "dev.label"))

if __name__ == "__main__":
    main()