mkdir -p logs/shakespeare_char
START_TIME=`date +%Y%m%d-%H:%M:%S`

# python train.py config/train_shakespeare_char.py
# python train.py config/train_shakespeare_char.py
# python train.py config/train_gpt2.py --wandb_log=False

# python sample.py --out_dir=out-shakespeare-char

# python train_transnormer.py config_transnormer/train_shakespeare_char.py #--wandb_log=True
# python train_transnormer.py config_transnormer/train_transnormer_small.py --wandb_log=False

# python sample_transnormer.py --out_dir=out-shakespeare-char-transnormer
# python sample.py --out_dir=out-shakespeare-char

torchrun --standalone --nproc_per_node=2 train_transnormer.py config_transnormer/train_transnormer_small.py