# You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
CUDA_VISIBLE_DEVICES=1 python main.py something RGB \
     --arch Mresnet18_ada --num_segments 8 \
     --gd 200 --lr 0.01 --wd 5e-4 --lr_steps 30 40 --epochs 50 \
     --batch-size 32 --iter_size 1 -j 4 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift_div=8 --shift --shift_place=blockres --npb  -scale 1  --pretrain=imagenet\
     -where2insert 0111 -insert_propotion all -layer_info S1 S1 S1 S1 -num_recur 4 -interacter PWI2_STPool_CAttendByAdd_outReLU_leaky11 -with_bn --mixed
     #--suffix TestSpeed #--evaluate --resume "checkpoint/TSM_something_RGB_Mresnet18_ada_shift8_blockres_avg_segment8_e50_scale1.0_mixed_0111-all-[['S', '1'], ['S', '1'], ['S', '1'], ['S', '1']]-PWI2_STPool_CAttendByAdd_outReLU-True-None_Tuesday_26_January_2021_21h_29m_58s/ckpt.best.pth.tar"
