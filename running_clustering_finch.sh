▽
export PYTHONPATH=$PYTHONPATH:$pwd
srun -u --partition=innova --job-name=f0.9 \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python  main.py --data_dir /mnt/lustre/ganxin/fedreid_data/data --train_all --clustering --clustering_method finch --max_distance 0.9 --resume_epoch 119 | tee fed_reid_clustering_cpk_finch_0.9_from_119.log &