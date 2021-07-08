mkdir -p log
now=$(date +"%Y%m%d_%H%M")

root_dir=/mnt/lustre/$(whoami)
project_dir=$root_dir/projects/FedReID
data_dir=$root_dir/fedreid_data

export PYTHONPATH=$PYTHONPATH:${project_dir}

srun -u --partition=innova --job-name=FedReID300 \
    -n8 --gres=gpu:8 --ntasks-per-node=8 \
    python ${project_dir}/main.py --data_dir $data_dir --rounds 300 \
    --datasets "Market-1501,DukeMTMC-reID,cuhk03,cuhk01,MSMT17,viper,prid,3dpes,ilids"  2>&1 | tee log/fedreid_${now}.log &