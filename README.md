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

> **🎉 We are now releasing the processed datasets.** (April, 2022)
>
> Please [email us](weiming001@e.ntu.edu.sg) (cc [this email](wingalong@gmail.com) in case it is filtered as spam) to request for the datasets with:
> 1. A short self-introduction.
> 2. The purposes of using these datasets.
>
> *⚠️ Further distribution of the datasets are prohibited.*

**We use 9 popular ReID datasets for the benchmark.**
<img src="images/datasets.png" width="700">


### Dataset Preprocess From Scratch

You can obtain the datasets from [awesome-reid-dataset](https://github.com/NEU-Gou/awesome-reid-dataset)

Dataset folder structure after preprocessing is provided [here](data_preprocess/README.md)

You can follow the following steps to preprocess datasets:

1. Download all datasets to `data_preprocess/data` folder. 
2. We provide the Json files for spliting the small datasets. (We haven't officially release the `split.json` files. Please send an email with short introduction to request for them.)
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
@inproceedings{zhuang2020performance,
  title={Performance Optimization of Federated Person Re-identification via Benchmark Analysis},
  author={Zhuang, Weiming and Wen, Yonggang and Zhang, Xuesen and Gan, Xin and Yin, Daiying and Zhou, Dongzhan and Zhang, Shuai and Yi, Shuai},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={955--963},
  year={2020}
}
```

## Maintainers
* Weiming Zhuang, Nanyang Technological University. [:octocat:](https://github.com/weimingwill)
* Xin Gan, Nanyang Technological University. [:octocat:](https://github.com/codergan)
* Daiying Yin, Nanyang Technological University. (Contributor)
