# BSc-thesis-ranking

My thesis is about logging policy induced relevance confounding.

## Models:
    RQ 1: Logging Policy Estimation (LPE)
        I) LambdaMART
        II) UPE confounder
    RQ 2: ULTR
        I) Two towers
            a) bare version
            b) with dropout FFNN bias tower
            c) UPE bias tower
        II) Best model from RQ 1
        III) LambdaMART directly trained on expert annotations (expert_rank.ipynb)
        IV) Naive model