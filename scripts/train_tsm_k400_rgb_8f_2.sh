# You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
CUDA_VISIBLE_DEVICES=0,1 python main_k400.py kinetics RGB \
     --arch Mresnet50_ada --num_segments 8 \
     --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 30 60 --epochs 75 \
     --batch-size 96 --iter_size 1 -j 8 --dropout 0.5 --consensus_type=avg --dense_sample --eval-freq=1 \
     --shift_div=8 --shift --shift_place=blockres --npb  -scale 1  --pretrain=imagenet\
     -where2insert 0011 -insert_propotion all -layer_info S1 S1 S1 S1 -num_recur 4 -interacter PWI1_pool_CAttendByAdd_KeepSpatial -with_bn --mixed \
     --suffix nearestUp_SGD_CustomDecay0_t1
     #--suffix nearestUp_eval_checkGrad --evaluate 
