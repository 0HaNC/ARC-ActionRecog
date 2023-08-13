# You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
CUDA_VISIBLE_DEVICES=1 python main.py diving48 RGB \
     --arch resnet101 --num_segments 16 \
     --gd 20 --lr 0.01 --wd 1e-4 --epochs 40 --warmup 10 --lr_type GradualWarmupCosine\
     --batch-size 8 --iter_size 1 -j 4 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift_div=8  --shift --shift_place=blockres --npb  -scale 1  --pretrain=imagenet --mixed \
     -where2insert 0000 -with_bn --suffix Ba8
     #--resume "checkpoint/TSM_something_RGB_Mresnet18_ada_avg_segment8_e60_scale1.0_mixed_1111-part-[['S', '4'], ['T', '1'], ['S', '2'], ['T', '1']]-donothing-True-None_Friday_27_November_2020_16h_52m_24s/ckpt.pth.tar"
