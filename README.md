# FedReID
Code for ACMMM 2020 oral paper - **[Performance Optimization for Federated Person Re-identification via Benchmark Analysis](https://arxiv.org/abs/2008.11560)**

Personal re-identification is an important computer vision task, but its development is constrained by the increasing privacy concerns. Federated learning is a privacy-preserving machine learning technique that learns a shared model across decentralized clients. In this work, we implement federated learning to person re-identification (**FedReID**) and optimize its performance affected by **statistical heterogeneity** in the real-world scenario. 

Algorithm: Federated Partial Averaging (FedPav)

<img src="images/fedpav-new.png" width="700">

## Prerequisite
* Install the libraries listed in requirements.txt
    ```
    pip install -r requirements.txt
    ```

## Datasets preparation
**We use 9 popular ReID datasets for the benchmark.**
<img src="images/datasets.png" width="700">



You can obtain the datasets from [awesome-reid-dataset](https://github.com/NEU-Gou/awesome-reid-dataset)

Dataset folder structure after preprocessing is provided [here](data_preprocess/README.md)

You can follow the following steps to preprocess datasets:

1. Download all datasets to `data_preprocess/data` folder. 
2. We provide the Json files for spliting the small datasets.
3. Run the following script to prepare all datasets:
    ```
    python prepare_all_datasets.py
    ```
4. Move the `data` folder to the root directory.
    ```
    move data_preprocess/data ./
    ```
5. For federated-by-identity scenario:
    ```
    python split_id_data.py
    ```
6. For federated-by-camera scenario:
    ```
    python split_camera_data.py
    ```
7. For merging all datasets to do merge training, you can use `rename_dataset.py` and `mix_datasets.py`.


## Run the experiments
Remember to save the log file for later use!
* Run Federated Partial Averaging (FedPav): 
    ```
    python main.py
    ```
* Run FedPav with knowledge distillation (KD): 
    ```
    python main.py --kd --regularization
    ```
* Run FedPav with cosine distance weight (CDW): 
    ```
    python main.py --cdw
    ```
* Run FedPav with knowledge distillation and cosine distance weight: 
    ```
    python main.py --cdw --kd --regularization
    ```

    
## Citation
```
@inproceedings{zhuang2020fedreid,
    author = {Zhuang, Weiming and Wen, Yonggang and Zhang, Xuesen and Gan, Xin and Yin, Daiying and Zhou, Dongzhan and Zhang, Shuai and Yi, Shuai},
    title = {Performance Optimization of Federated Person Re-Identification via Benchmark Analysis},
    year = {2020},
    isbn = {9781450379885},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3394171.3413814},
    doi = {10.1145/3394171.3413814},
    booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
    pages = {955–963},
    numpages = {9},
    keywords = {machine learning system, computer vision, federated learning},
    location = {Seattle, WA, USA},
    series = {MM '20}
}
```
