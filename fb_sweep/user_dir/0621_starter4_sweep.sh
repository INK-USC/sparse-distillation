cd ..

python ./fb_sweep/0621_starter4_sweep.py \
    --data /data/home/yeqy/src/fairseq/starter1/data/IMDB-trigram-bin/ \
    --prefix imdb_trigram \
    --num-trials 32 \
    --num-gpus 1 \
    --partition a100 \
    --backend slurm \
    --checkpoints-dir /data/home/yeqy/sweep/2021-06-21/

# python ./fb_sweep/agg_results.py "/data/home/yeqy/sweep/2021-06-21/**/train.log" \
#     --log-pattern valid \
#     --keep_cols loss,accuracy \
#     --sort_col accuracy