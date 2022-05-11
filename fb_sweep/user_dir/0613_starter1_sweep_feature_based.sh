cd ..

python ./fb_sweep/starter1_sweep_feature_based.py \
    --data /data/home/yeqy/src/fairseq/starter1/data/IMDB-trigram-bin/ \
    --prefix imdb_trigram \
    --num-trials -1 \
    --num-gpus 1 \
    --partition a100 \
    --backend slurm \
    --checkpoints-dir /fsx/yeqy/sweep/2021-06-13/

    # --backend slurm \

# python ./fb_sweep/agg_results.py "/fsx/yeqy/sweep/2021-06-13/**/train.log" \
#     --log-pattern valid \
#     --keep_cols loss,accuracy \
#     --sort_col accuracy