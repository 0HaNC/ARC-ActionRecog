# You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
CUDA_VISIBLE_DEVICES=1 python main_k400.py kinetics RGB \
     --arch resnet18 --num_segments 8 \
     --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 45 65 --epochs 75 \
     --batch-size 64 --iter_size 1 -j 0 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --mixed --pretrain=imagenet --suffix center_1 --evaluate
