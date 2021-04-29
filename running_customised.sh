export PYTHONPATH=$PYTHONPATH:$pwd
srun -u --partition=irdcRD --job-name=bs${1}_le${2}_time${3} \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python  main.py --data_dir /mnt/lustre/ganxin/fedreid_data/data --train_all --local_epoch ${1} --batch_size ${2} --experiment_index ${3}| tee fed_reid_base_bs${1}_le${2}_time${3}.log &