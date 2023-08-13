# You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
CUDA_VISIBLE_DEVICES=0 python main.py something RGB \
     --arch resnet34 --num_segments 8 \
     --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 25 40 --epochs 50 \
     --batch-size 16 -j 4 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift_div=8  --shift_place=naive_skip_connection --npb  -scale 1 --group 1 --pretrain=imagenet --gpus 0 --shift  #--mixed #--suffix torch1.6 #--suffix checkGrad #--tune_from ./pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth #--gpus 1 #--shift_place=blockres --pretrain=no
