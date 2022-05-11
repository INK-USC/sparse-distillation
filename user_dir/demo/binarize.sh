cd ..

SRC_DIR=/path/to/files/with/ngram/indices
DEST_DIR=/path/to/save/the/binarized/dataset
BIN_DIR=/path/to/data-bin/when/finetune/roberta

# a hack: we use the IMDB train set here in --testpref and rename it later.
# we will be monitoring the performance on IMDB train set through out the distillation process.

fairseq-preprocess \
    --only-source \
    --trainpref "${SRC_DIR}/train.trigram.input0" \
    --validpref "${SRC_DIR}/dev.trigram.input0" \
    --testpref "${SRC_DIR}/train.trigram.original.input0" \
    --destdir "${DEST_DIR}/input0" \
    --workers 50

fairseq-preprocess \
    --only-source \
    --trainpref "${SRC_DIR}/train.label" \
    --validpref "${SRC_DIR}/dev.label" \
    --testpref "${SRC_DIR}/train.original.label" \
    --destdir "${DEST_DIR}/label" \
    --srcdict "${BIN_DIR}/label/dict.txt" \
    --workers 50

# change test set name to valt
mv ${DEST_DIR}/input0/test.idx ${DEST_DIR}/input0/valt.idx 
mv ${DEST_DIR}/input0/test.bin ${DEST_DIR}/input0/valt.bin
mv ${DEST_DIR}/label/test.idx ${DEST_DIR}/label/valt.idx 
mv ${DEST_DIR}/label/test.bin ${DEST_DIR}/label/valt.bin 