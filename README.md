# BiVAE with skip network decoder

```c
.
├── checkout
│   ├── metrics_log.txt
│   ├── model  // model weight saving path
│   ├── params // model params log
│   └── recommend // recommend result of model
├── config
│   └──  config.py
├── data  // data folder
│   ├── cs608_ip_train_v3.csv // sample raw data
│   └── Modify_train_4_25.csv  //sample training data
├── data_process.sh
├── gen_data.py
├── img  // readme file img
├── models
│   ├── Modify_bivaecf.py
│   ├── ModifyBivae.py
│   └── modifybpr.py
├── README.md
├── negative_sampling.md  // negative sampling detail
├── train_bivae.py
├── train_mbpr.py
├── train_mbpr.sh
└── utils
    ├── cornac_utils.py
    ├── metrics.py
    └── processer.py
```

## About This Repo

This repository includes the following two ideas to improve BPR and [Bivae](https://cornac.readthedocs.io/en/latest/_modules/cornac/models/bivaecf/recom_bivaecf.html#BiVAECF):

1. Negative sampling are tried for the BPR recommender, however, this model **do not** outperform [cornac BPR](https://github.com/PreferredAI/cornac) in the test. The detail of negative sampling method is describe [here](negative_sampling.md).
2. Since for this task, we do not have user and item side information. Therefore it is hard to generate a cap prior using VAE. To solve the prior collapse problem, [skip VAE](https://arxiv.org/abs/1807.04863) was tried. 

The model was modified from [cornac BiVAE](https://cornac.readthedocs.io/en/latest/_modules/cornac/models/bivaecf/recom_bivaecf.html#BiVAECF). **Study and analysis purpose only.**

Detail of BiVAE mechanism please referred to my blog: [source code analysis for cornac BiVAE](http://wujiawen.xyz/2021/06/07/BiVAE/).

| Model                               | harmonic_mean | NDCG       | NCRR       | RECALL     |
| ----------------------------------- | ------------- | ---------- | ---------- | ---------- |
| BPR                                 | 0.0406        | 0.0457     | 0.0223     | 0.1295     |
| negative sampling                   | 0.0030        | 0.0043     | 0.0014     | 0.014      |
| BiVAE                               | 0.0477        | 0.0507     | 0.0285     | 0.1303     |
| BiVAE with decoder applied skip VAE | 0.0502        | **0.0531** | **0.0297** | **0.1366** |

*(table: model scores.)*

The negative sampling performs bad might owning to the following reason:

+ Negative sample selecting criterion is not optimized. In this project, the selected weight of negative sample has not been tuned.
+ Number of negative sample is not optimized. In this project,  only new `num_neg` values (2, 5, 10, 15, 25) is tested.  `num_neg` = 25 usually yields better.
+ **Poor training speed.** Since the code is implement in python and numpy only, it is expected to run much slower. This is also the reason that few hyper parameters are tuned in this method.

## How to use

#### Training BiVAE and generate top 50 recommend

```shell
python train_bivae.py --epochs 850 --work_path "." --true_threshold 0 --lr 0.001 --k 32 --encoder_structure 64 --decoder_structure 64 --batch_size 128 --seed 7
```

where encoder structure and decoder structure will be `[64] ` and `[64, 64]` if you want multiple layer decoder structure just pass `--decoder_structure 64 64 64`

**Comparing with cornac BiVAE.**

```python
# other parameters setting:
activation = 'tanh'
likelihood = "pois"
beta kl = 1
batch size = 128
seed = 7
learning_rate = 1e-3
```

| Encoder structure | Decoder structure (Skip VAE only) | K    | BiVAE (NDCG， recall) | BiVAE with skip network decoder |
| ----------------- | --------------------------------- | ---- | --------------------- | ------------------------------- |
| 128               | [128]                             | 64   | 0.0344,  **0.0892**   | **0.0354**, 0.0885              |
| 64                | [64]                              | 32   | 0.038,  0.0945        | **0.0392, 0.0973**              |
| 40                | [40]                              | 20   | 0.0358, **0.0955**    | **0.0369**, 0.0926              |

*(table: NCDG, recall scores.)*

#### Processing Negative Sample

```shell
python gen_data.py --num_neg 5 --true_threshold 2 --work_path "." --verbose --dtype "merge" --raw_data_path '/data/cs608_ip_merge_v3.csv'
```

Output file will be named according to:`--dtype --num_neg --true_threshold`  

Or you could modify `data_process.sh` to generate many processed datasets for training. Change argument `--raw_data_path` to the data path of your raw data, which is `\data\cs608_ip_train_v3.csv` in default.

#### Training BPR

you can train your model using:

```python
python train.py --lr 0.1 --k 128 --lambda_reg 0.003 --epochs 200 --train_data_path "/data/Modify_train_4_20.csv" --true_threshold 4 --num_neg 20 --work_path "."
```

or just run `. ./train.sh`

Model will save according to: `--true_threshold t --num_neg n --k k`.

**Model Loading and Saving for modified BPR model**

```python
from model.modifybpr import MBPR
model = MBPR(*args)

# to load a saved parameter, checkout_path default "/checkout"
# model will load weight from checkout_path/model/
model.load(checkout_path)  

# fit a model
model.fit(train_set, num_users, num_items)  

# save model weight to csv file. checkout_path default "/checkout" model weight will be saved to checkout_path/model/
model.save(checkout_path) 
```



**Since the model is implement based on python and numpy, you should expected slower optimizing speed than other models.**





