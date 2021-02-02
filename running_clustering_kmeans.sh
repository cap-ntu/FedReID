export PYTHONPATH=$PYTHONPATH:$pwd
srun -u --partition=innova --job-name=kmeans_2 \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python  main.py --data_dir /mnt/lustre/ganxin/fedreid_data/data --train_all --clustering --clustering_method kmeans --n_cluster 2 --resume_epoch 149 | tee fed_reid_clustering_cpk_kmeans_2_from_149.log &