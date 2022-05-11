cd ..

python ./fb_sweep/0616_starter1_sweep_bsz.py \
    --data /data/home/yeqy/src/fairseq/starter1/data/IMDB-trigram-bin/ \
    --prefix imdb_trigram \
    --num-trials -1 \
    --num-gpus 1 \
    --partition a100 \
    --backend slurm \
    --checkpoints-dir /data/home/yeqy/sweep/2021-06-16/

    # --backend slurm \

python ./fb_sweep/agg_results.py "/data/home/yeqy/sweep/2021-06-16/**/train.log" \
    --log-pattern valid \
    --keep_cols loss,accuracy \
    --sort_col accuracy