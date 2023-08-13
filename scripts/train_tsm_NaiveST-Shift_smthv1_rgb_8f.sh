# You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
CUDA_VISIBLE_DEVICES=1 python main.py something RGB \
     --arch pwresnet50 --num_segments 8 \
     --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 25 40 --epochs 55 \
     --batch-size 16 -j 3 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift_div=8  --shift_place=blockres --npb  -scale 1 --groups 4 --pretrain=no --gpus 0 --shift --mixed --suffix naiveSTshift #--mixed #--suffix torch1.6 #--suffix checkGrad #--tune_from ./pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth #--gpus 1 #--shift_place=blockres --pretrain=no
