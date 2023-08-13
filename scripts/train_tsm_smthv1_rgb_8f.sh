# You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
CUDA_VISIBLE_DEVICES=0 python main.py something RGB \
     --arch resnet18 --num_segments 8 \
     --gd 200 --lr 0.01 --iter_size 1 --wd 5e-4 --lr_steps 20 30 --epochs 40 \
     --batch-size 8 -j 4 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift_div=8 --shift --shift_place=blockres --npb  -scale 1 --group 1 --pretrain=NO --gpus 0  --mixed --suffix NOPRETRAIN #--evaluate #--non_local #--suffix eva --evaluate
      #--suffix checkGrad #--tune_from ./pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth #--gpus 1 #--shift_place=blockres --pretrain=no
