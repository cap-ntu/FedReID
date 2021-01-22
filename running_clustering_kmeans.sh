export PYTHONPATH=$PYTHONPATH:$pwd
srun -u --partition=Sensetime --job-name=clu_kmeans_2 \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python  main.py --data_dir /mnt/lustre/ganxin/fedreid_data/data --train_all --clustering --clustering_method kmeans --n_cluster 2 | tee fed_reid_clustering_cpk_kmeans_2.log &