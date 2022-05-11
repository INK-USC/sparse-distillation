cd ..

python ./fb_sweep/0623_starter4_sweep_downstream.py \
    --data /data/home/yeqy/src/fairseq/starter3/data/imdb-n-amazon-all-smalldict-original-bin/ \
    --prefix imdb_trigram \
    --num-trials 32 \
    --num-gpus 1 \
    --partition a100 \
    --backend slurm \
    --checkpoints-dir /data/home/yeqy/sweep/2021-06-23/

# python ./fb_sweep/agg_results.py "/data/home/yeqy/sweep/2021-06-23/**/train.log" \
#     --log-pattern valid \
#     --keep_cols loss,accuracy \
#     --sort_col accuracy