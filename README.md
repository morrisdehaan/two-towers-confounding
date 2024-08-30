# Two towers confounding

Code from the paper "Understanding the Effects of the Baidu-ULTR Logging Policy on Two Tower Models"

## Models:
    RQ 1: Logging Policy Estimation (LPE)
        I) LambdaMART
        II) UPE confounder
    RQ 2: ULTR
        I) Two towers
            a) FFNN bias tower
            b) dropout added to dropout FFNN bias tower
            c) UPE bias tower
        II) Best model from RQ 1
        III) LambdaMART directly trained on expert annotations
        IV) Naive model (relevance tower without bias modelling)

## Usage
Fit or use a pretrained model by running:
```
python train.py config.json params_dir
```

This trains and evaluates the model from the configuration file and loads or stores fitted parameters in `res/params_dir`.
