export PYTHONPATH=$PYTHONPATH:$pwd
srun -u --partition=Sensetime --job-name=clu_cdw_kd \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python  main.py --data_dir /mnt/lustre/ganxin/fedreid_data/data --train_all --clustering --cdw --kd --regularization --kd_method whole | tee fed_reid_clustering_cdw_kd_whole.log &