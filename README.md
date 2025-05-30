# CS3308 Machine Learning Homework

Based on the [Crossformer Codebase](https://github.com/Thinklab-SJTU/Crossformer)

My codebase [ML-HW](https://github.com/jason-fxz/ML-HW)

branch [paper](https://github.com/jason-fxz/ML-HW/tree/paper) for paper implementation

## Modifications

- `cross_models/RevIN.py` for RevIN implementation
- `utils/weekly_pattern_aligner` for WPRL implementation
- some modifications in `main_crossformer.py`,`exp_crossformer.py`,`data_loader.py` to support RevIN and WPRL

## Run experiments

Example commands to run experiments on the ECL dataset using CrossFormer model with different configurations:

```bash
# baseline 
python main_crossformer.py --data ECL \
--in_len 720 --out_len 720 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-5  --lradj fixed  --itr 3

# RevIN 
python main_crossformer.py --data ECL \
--in_len 720 --out_len 720 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-5  --lradj fixed  --itr 3 --use_revin --tag RevIN

# WPRL (ours)
python main_crossformer.py --data ECL \
--in_len 720 --out_len 720 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-5  --lradj fixed  --itr 3 --use_weekly_pattern --tag WPRL
```
