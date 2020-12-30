export PYTHONPATH=$PYTHONPATH:$pwd
srun -u --partition=Sensetime --job-name=fed_clustering \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python  main.py | tee fed_clustering.log &