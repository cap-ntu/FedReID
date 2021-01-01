export PYTHONPATH=$PYTHONPATH:$pwd
srun -u --partition=Sensetime --job-name=fed_clustering \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python  main.py --data_dir /mnt/lustre/ganxin/fedreid_data/data --train_all | tee fed_clustering_base.log &