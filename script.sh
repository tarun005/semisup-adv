python robust_self_training.py --dataset svhn --data_dir /newfoundland/tarun/datasets/Digits/SVHN/ --model wrn-40-2 --batch_size 300 --test_batch_size 512 --epochs 120 --eval_freq 4 --model_dir svhn_aug_RST_adv --distance l_inf --epsilon 0.031 --svhn_extra  | tee outputs/svhn_aug.out
