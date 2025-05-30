## ECL

############################predict length 336####################################

# baseline
python main_crossformer.py --data ECL \
--in_len 168 --out_len 336 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-4  --lradj fixed --itr 3 

# +RevIN
python main_crossformer.py --data ECL \
--in_len 168 --out_len 336 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-4  --lradj fixed --itr 3 --use_revin --tag RevIN 

# +WPRL (ours)
python main_crossformer.py --data ECL \
--in_len 168 --out_len 336 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-4  --lradj fixed --itr 3 --use_weekly_pattern --tag WPRL

############################predict length 720####################################

# baseline
python main_crossformer.py --data ECL \
--in_len 720 --out_len 720 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-5  --lradj fixed  --itr 3

# +RevIN
python main_crossformer.py --data ECL \
--in_len 720 --out_len 720 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-5  --lradj fixed  --itr 3 --use_revin --tag RevIN

# +WPRL (ours)
python main_crossformer.py --data ECL \
--in_len 720 --out_len 720 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-5  --lradj fixed  --itr 3 --use_weekly_pattern --tag WPRL

############################predict length 960####################################

# baseline
python main_crossformer.py --data ECL \
--in_len 720 --out_len 960 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-5  --lradj fixed  --itr 3

# +RevIN
python main_crossformer.py --data ECL \
--in_len 720 --out_len 960 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-5  --lradj fixed  --itr 3 --use_revin --tag RevIN

# +WPRL (ours)
python main_crossformer.py --data ECL \
--in_len 720 --out_len 960 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-5  --lradj fixed  --itr 3 --use_weekly_pattern --tag WPRL
