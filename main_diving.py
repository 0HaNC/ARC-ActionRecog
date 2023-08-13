# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms_lab import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter
import pdb
from ipdb import set_trace
from datetime import datetime
from functools import partial
best_prec1 = 0
import math
import shutil
import re
print(torch.cuda.device_count())
#pdb.set_trace()


def temperature_curve(x):
    return 1
    return 1 * 10**(-x/15)

def main():
    #torch.autograd.set_detect_anomaly(True)
    global args, best_prec1
    args = parser.parse_args()

    if args.dist:
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)

    mixed = args.mixed
    # set GradScaler
    from torch.cuda.amp import GradScaler
    scaler = GradScaler(enabled=mixed)
    # set autocast
    if mixed:
        from torch.cuda.amp import autocast
    else:
        class dummy:
            def __enter__(self): pass
            def __exit__(self,typ,value,trace): pass
        autocast = dummy

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    args.store_name += '_scale{}'.format(args.scale)
    if args.groups!=1:
        args.store_name += '_group{}'.format(args.groups)
    if args.shuffle:
        args.store_name += '_shuffle'
    if args.shift_types:
        args.store_name += '_shiftType-{}'.format(args.shift_types)

    if args.mixed:
        args.store_name += '_mixed'
    if args.dist:
        args.store_name += '_dist'
    args.store_name += '_{where2insert}-{insert_propotion}-{layer_info}-{interacter}-{with_bn}-{suffix}'.format(
        layer_info=args.layer_info, where2insert=args.where2insert, 
        insert_propotion=args.insert_propotion, interacter=args.interacter, with_bn=args.with_bn, suffix=args.suffix)
    args.store_name = args.store_name + '_{time}'.format(time=datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss'))
    if args.resume:
        args.store_name += '_resume'    

    print('storing name: ' + args.store_name)

    check_rootfolders()
    shutil.copyfile('archs/ARC.py', 'archs/ARC.py'.replace('archs', os.path.join('log', args.store_name)))
    shutil.copyfile('archs/MultiBranch_HWT_adaptive_Resnet.py', 'archs/MultiBranch_HWT_adaptive_Resnet.py'.replace('archs', os.path.join('log', args.store_name)))

    if 'something' in args.dataset:
        # label transformation for left/right categories
        # please refer to labels.json file in sometingv2 for detail.
        target_transforms = {86:87,87:86,93:94,94:93,166:167,167:166}
    else:
        target_transforms = None

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local, groups=args.groups, shuffle=args.shuffle, args=args, target_transforms = target_transforms)
    #set_trace()
    if 'ada' in args.arch:
        grad_scale = True
    grad_scale = False
    if grad_scale:
        from functools import partial
        def scale_grad_hook(module, grad_input, grad_output, scale, name):
            #set_trace()
            #print('haha')
            return (grad_input[0]/scale,)
        num_blocks = ['111','0111','011111','011']
        stages = [model.base_model.layer1, model.base_model.layer2, model.base_model.layer3, model.base_model.layer4]
        for k, stage in enumerate(num_blocks):
            for i, b in enumerate(stage):
                if b=='0': continue
                get_attr_name = str(i)
                block = getattr(stages[k], get_attr_name)
                SCALE = 4
                #set_trace()
                for j, s in enumerate(range(SCALE, 1, -1)):
                    block.conv2.layer[j].register_backward_hook(partial(scale_grad_hook, scale=s, name='BP_subconv1'))
                #set_trace()
                setattr(stages[k], get_attr_name, block)

    if args.suffix and 'checkGrad' in args.suffix:
        from functools import partial
        mode = 'minmax'
        def backward_hook(module, grad_in, grad_out, index):
            if index==1:
                if mode=='avg':
                    grad_block1.append(grad_out[0].abs().mean().detach().cpu().numpy())
                elif mode=='minmax':
                    grad_block1.append([grad_out[0].abs().min().detach().cpu().item(), grad_out[0].abs().max().detach().cpu().item()])
                else:
                    raise NotImplementedError
            if index==2:
                if mode=='avg':
                    grad_block2.append(grad_out[0].abs().mean().detach().cpu().numpy())
                elif mode=='minmax':
                    grad_block2.append([grad_out[0].abs().min().detach().cpu().item(), grad_out[0].abs().max().detach().cpu().item()])
            if index==3:
                if mode=='avg':
                    grad_block3.append(grad_out[0].abs().mean().detach().cpu().numpy())
                elif mode=='minmax':
                    grad_block3.append([grad_out[0].abs().min().detach().cpu().item(), grad_out[0].abs().max().detach().cpu().item()])
        def print_max_grad(module, grad_in, grad_out, name=''):
            print(name, grad_out[0].abs().max().detach().cpu().item(), grad_in[0].abs().max().detach().cpu().item(), grad_in[0].size(), grad_in[0].size())
        def print_norm_grad(module, grad_in, grad_out, name=''):
            #set_trace()
            if grad_in[0] is not None:
                print(name, grad_out[0].norm().detach().cpu().item(), grad_in[0].norm().detach().cpu().item(), grad_out[0].size(), grad_in[0].size())
            else:
                print(name, grad_out[0].norm().detach().cpu().item(), grad_out[0].size())
        def print_full_grad(module, grad_in, grad_out, name=''):
            print(name, grad_out[0].detach().cpu(), grad_in[0].detach().cpu())

        def backward_store_hook(module, grad_in, grad_out, name):
            grad_block1.append((grad_in[0].detach().cpu(), grad_out[0].detach().cpu()))
        def forward_store_hook(module, input, output, name):
            store_block.append((input[0].detach().cpu(), output[0].detach().cpu()))
        if 'resnet' in args.arch:
            model.base_model.layer4.register_forward_hook(partial(forward_store_hook, name='L4'))
            '''
            model.base_model.conv1.register_backward_hook(partial(backward_hook, index=1))
            model.base_model.layer2.register_backward_hook(partial(backward_hook, index=2))
            model.base_model.layer4.register_backward_hook(partial(backward_hook, index=3))
            '''
            
            #block.conv2.softmax.register_backward_hook(partial(print_grad_, name='BP_softmax1'))
            #block.conv2.layer[0].register_forward_hook(partial(forward_store_hook, name='FW_subconv0'))
            #block.conv2.Weight.register_hook(lambda grad:print('grads of Weight', grad))
            '''
            block = getattr(model.base_model.layer1, '0')
            block.bn2.register_backward_hook(partial(print_norm_grad, name='Layer1.0_Conv2_BP'))
            setattr(model.base_model.layer1, '0', block)
            block = getattr(model.base_model.layer1, '1')
            block.bn2.register_backward_hook(partial(print_norm_grad, name='Layer1.1_Conv2_BP'))
            setattr(model.base_model.layer1, '1', block)
            block = getattr(model.base_model.layer1, '2')
            block.bn2.register_backward_hook(partial(print_norm_grad, name='Layer1.2_Conv2_BP'))
            setattr(model.base_model.layer1, '2', block)
            block = getattr(model.base_model.layer1, '3')
            block.bn2.register_backward_hook(partial(print_norm_grad, name='Layer1.3_Conv2_BP'))
            setattr(model.base_model.layer1, '3', block)
            '''
            '''
            model.base_model.conv1.register_backward_hook(partial(print_norm_grad, name='Conv1_BP'))
            model.base_model.bn1.register_backward_hook(partial(print_norm_grad, name='BN1_BP'))
            #
            block = getattr(model.base_model.layer1, '0')
            block.conv1.register_backward_hook(partial(print_norm_grad, name='Layer1.0.conv1_BP'))
            setattr(model.base_model.layer1, '0', block)
            block = getattr(model.base_model.layer1, '2')
            block.bn3.register_backward_hook(partial(print_norm_grad, name='Layer1.2.bn3_BP'))
            setattr(model.base_model.layer1, '2', block)
            #
            block = getattr(model.base_model.layer2, '0')
            block.conv1.register_backward_hook(partial(print_norm_grad, name='Layer2.0.conv1_BP'))
            setattr(model.base_model.layer2, '0', block)
            block = getattr(model.base_model.layer2, '3')
            block.bn3.register_backward_hook(partial(print_norm_grad, name='Layer2.3.bn3_BP'))
            setattr(model.base_model.layer2, '3', block)
            #
            block = getattr(model.base_model.layer3, '0')
            block.conv1.register_backward_hook(partial(print_norm_grad, name='Layer3.0.conv1_BP'))
            setattr(model.base_model.layer3, '0', block)
            block = getattr(model.base_model.layer3, '5')
            block.bn3.register_backward_hook(partial(print_norm_grad, name='Layer3.5.bn3_BP'))
            setattr(model.base_model.layer3, '5', block)
            #
            block = getattr(model.base_model.layer4, '0')
            block.conv1.register_backward_hook(partial(print_norm_grad, name='Layer4.0.conv1_BP'))
            setattr(model.base_model.layer4, '0', block)
            block = getattr(model.base_model.layer4, '2')
            block.bn3.register_backward_hook(partial(print_norm_grad, name='Layer4.2.bn3_BP'))
            setattr(model.base_model.layer4, '2', block)
            '''


        elif 'Inception' in args.arch:
            model.base_model.pool1_3x3_s2.register_backward_hook(partial(backward_hook, index=1))
            model.base_model.inception_3c_pool.register_backward_hook(partial(backward_hook, index=2))
            model.base_model.inception_5b_relu_pool_proj.register_backward_hook(partial(backward_hook, index=3))
        else:
            raise NotImplementedError

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    #set_trace()
    if 'jester' in args.dataset:
        raise ValueError('flip should be disabled when using jester dataset')
    train_augmentation = model.get_augmentation(flip=True)

    #model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    if args.dist:
        model = model.cuda()
        model = nn.parallel.DistributedDataParallel(model)
    else:
        #model = model.to(args.gpus[0])
        model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    
    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
                                
    #set_trace()
    #optimizer = torch.optim.Adam(policies, args.lr, weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        #set_trace()
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))

        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        #import pdb;pdb.set_trace()
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        '''
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))
        #import pdb;pdb.set_trace()
        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        #import pdb;pdb.set_trace()
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)
        '''
        '''
        replace_dict = []
        for k, v in sd.items():
            if 'module.'in k:
                replace_dict.append(k)
        for k in replace_dict:
            sd[k[len('module.'):]] = sd.pop(k)
        model.load_state_dict(sd)
        '''
        if 'kinetics' in args.tune_from:
            del sd['module.new_fc.weight']
            del sd['module.new_fc.bias']
            model_dict = model.state_dict()
            model_dict.update(sd)
            model.load_state_dict(model_dict)
            del sd
        else:
            model.load_state_dict(sd)


    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True
    #torch.backends.cudnn.benchmark = True
    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.dataset, args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       GroupScale((240,320)),
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample, scale=args.scale),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.dataset, args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       #GroupScale(int(scale_size)),
                       GroupScale((240,320)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample, scale=args.scale, twice_sample=True, test_mode=True),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        '''
        with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as profile:
            for i, (input, target) in enumerate(val_loader):
                if i==3:
                    break
                with torch.no_grad():
                    y = model(input)
        print(profile)
        profile.export_chrome_trace('trace_ada-shift.json')
        '''
        validate(val_loader, model, criterion, 0, mixed=False, autocast=autocast)        
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps, it=0, iters_per_ep=len(train_loader))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer, mixed, autocast, scaler)
        #exit()
        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, epoch, log_training, tf_writer, mixed, autocast)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

            output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer, mixed, autocast, scaler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # Annealing
    T = torch.tensor(temperature_curve(epoch)).cuda()
    with torch.no_grad():
        check_keyword = 'channel_interacter.T'
        for name, param in model.named_parameters():
            if check_keyword in name:
                param.copy_(T)
    #pdb.set_trace()
    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        if args.lr_type=='GradualWarmupCosine' and epoch<args.warmup:
            adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps, it=i, iters_per_ep=len(train_loader))
        with autocast():
            # compute output
            #with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as profile:
            output = model(input_var)
            '''
            set_trace()
            print(profile)
            profile.export_chrome_trace('trace_tsm.json')
            '''
            loss = criterion(output, target_var) / args.iter_size

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        #set_trace()
        #print('\nLayer1 block1:', grad_block2[-1])
        #print('\nLayer1 block0:', grad_block1[-1])

        '''
        check_list = ['module.base_model.layer1.1.conv2.weight', 'module.base_model.layer3.1.conv2.weight']
        Dict = dict()
        for name, param in model.named_parameters():
            if name in check_list:
                print(name, param.grad)
        set_trace()
        '''
        if (i+1) % args.iter_size == 0:
            # scale down gradients when iter size is functioning
            if args.clip_gradient is not None:
                scaler.unscale_(optimizer)
                if args.suffix and 'StoreGradViaTensorboard' in args.suffix:
                    #set_trace()
                    save_freq = args.print_freq*10
                    if i % save_freq == 0:
                        tf_writer.add_histogram('Grads/layer1/sp1', model.module.base_model.layer1[0].conv2.channel_interacter.Weight_mat.grad[:, 0*16:1*16].cpu().detach(), (epoch*len(train_loader)+i)//save_freq)
                        tf_writer.add_histogram('Grads/layer1/sp2', model.module.base_model.layer1[0].conv2.channel_interacter.Weight_mat.grad[:, 1*16:2*16].cpu().detach(), (epoch*len(train_loader)+i)//save_freq)
                        tf_writer.add_histogram('Grads/layer1/sp3', model.module.base_model.layer1[0].conv2.channel_interacter.Weight_mat.grad[:,2*16:3*16].cpu().detach(), (epoch*len(train_loader)+i)//save_freq)
                        tf_writer.add_histogram('Grads/layer2/sp1', model.module.base_model.layer2[1].conv2.channel_interacter.Weight_mat.grad[:, 0*32:1*32].cpu().detach(), (epoch*len(train_loader)+i)//save_freq)
                        tf_writer.add_histogram('Grads/layer2/sp2', model.module.base_model.layer2[1].conv2.channel_interacter.Weight_mat.grad[:, 1*32:2*32].cpu().detach(), (epoch*len(train_loader)+i)//save_freq)
                        tf_writer.add_histogram('Grads/layer2/sp3', model.module.base_model.layer2[1].conv2.channel_interacter.Weight_mat.grad[:,2*32:3*32].cpu().detach(), (epoch*len(train_loader)+i)//save_freq)
                        tf_writer.add_histogram('Grads/layer4/sp1', model.module.base_model.layer4[1].conv2.channel_interacter.Weight_mat.grad[:, 0*128:1*128].cpu().detach(), (epoch*len(train_loader)+i)//save_freq)
                        tf_writer.add_histogram('Grads/layer4/sp2', model.module.base_model.layer4[1].conv2.channel_interacter.Weight_mat.grad[:, 1*128:2*128].cpu().detach(), (epoch*len(train_loader)+i)//save_freq)
                        tf_writer.add_histogram('Grads/layer4/sp3', model.module.base_model.layer4[1].conv2.channel_interacter.Weight_mat.grad[:,2*128:3*128].cpu().detach(), (epoch*len(train_loader)+i)//save_freq)
                        
                        '''
                        tf_writer.add_histogram('Grads/layer1', {
                            'sp1':model.module.base_model.layer1[0].conv2.channel_interacter.Weight_mat.grad[:, 0*16:1*16],
                            'sp2':model.module.base_model.layer1[0].conv2.channel_interacter.Weight_mat.grad[:, 1*16:2*16],
                            'sp2':model.module.base_model.layer1[0].conv2.channel_interacter.Weight_mat.grad[:, 2*16:3*16]},
                             (epoch*len(train_loader)+i)//save_freq)
                        tf_writer.add_histogram('Grads/layer2', {
                            'sp1':model.module.base_model.layer2[0].conv2.channel_interacter.Weight_mat.grad[:, 0*32:1*32],
                            'sp2':model.module.base_model.layer2[0].conv2.channel_interacter.Weight_mat.grad[:, 1*32:2*32],
                            'sp2':model.module.base_model.layer2[0].conv2.channel_interacter.Weight_mat.grad[:, 2*32:3*32]},
                             (epoch*len(train_loader)+i)//save_freq)
                        tf_writer.add_histogram('Grads/layer4', {
                            'sp1':model.module.base_model.layer4[0].conv2.channel_interacter.Weight_mat.grad[:, 0*128:1*128],
                            'sp2':model.module.base_model.layer4[0].conv2.channel_interacter.Weight_mat.grad[:, 1*128:2*128],
                            'sp2':model.module.base_model.layer4[0].conv2.channel_interacter.Weight_mat.grad[:, 2*128:3*128]},
                             (epoch*len(train_loader)+i)//save_freq)
                        '''

                total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
                if total_norm > args.clip_gradient:
                    print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))          
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            
            check_list = ['module.base_model.layer1.0.conv2.channel_interacter.Weight_mat', 'module.base_model.layer2.1.conv2.channel_interacter.Weight_mat', 'module.base_model.layer4.1.conv2.channel_interacter.Weight_mat',
            'module.base_model.layer2.1.conv2.channel_interacter.Weight_mat2',
            'module.base_model.layer2.1.conv2.channel_interacter.dw_weights',
            'module.base_model.layer2.1.conv2.channel_interacter.Weight_mat_add', 'module.base_model.layer2.1.conv2.channel_interacter.Weight_mat_add2',
            'module.base_model.layer2.1.conv2.channel_interacter.T', 
            'module.base_model.layer2.1.conv2.channel_interacter.w', 'module.base_model.layer2.1.conv2.channel_interacter.b',
            'module.base_model.layer2.1.conv2.channel_interacter.bns.2.weight',
            'module.base_model.layer2.1.conv1.affiner.w',
            ]
            if args.suffix and 'checkPretrian' in args.suffix:
                check_list+=['module.base_model.layer2.1.bn2.weight', 'module.base_model.layer2.1.conv1.layer.2.block.1.weight']
            if args.interacter and 'PWI1_bn' in args.interacter:
                check_list += ['module.base_model.layer2.1.conv2.channel_interacter.bn.weight', 'module.base_model.layer2.1.conv2.channel_interacter.bn.bias']
            for name, param in model.named_parameters():
                #print(name)
                if name in check_list or (re.search(r'.*.0.conv2.channel_interacter.Weight_mat*',name)):
                    if param.dim() != 0:
                        output = '{}: max:{:.06f} abs_mean:{:.06f} mean:{:.06f} min:{:.06f}'.format(name, param.detach().cpu().max().item(), 
                            param.detach().abs().cpu().mean().item(), param.detach().cpu().mean().item(), param.detach().cpu().min().item())
                        print(output)
                        if log is not None:
                            log.write(output + '\n')
                            log.flush()
                    else:
                        print(name, param.detach().cpu().item())
            #sd = model.state_dict()
            #set_trace()
            #print(sd['module.base_model.layer1.1.conv1.weight'], sd['module.base_model.layer3.1.conv1.weight'])
            #print(sd['module.base_model.layer1.1.conv2.weight'], sd['module.base_model.layer3.1.conv2.weight'])
            if args.suffix and 'checkGrad' in args.suffix:
                #print(grad_block1[-1], grad_block2[-1], grad_block3[-1])
                pass
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'scaled_Loss {scaled_Loss:.4f}\t'
                      'scale {scale:.1f}\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, scale=scaler.get_scale(), scaled_Loss=scaler.scale(loss).item(), top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()
        #exit()
    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
    if args.suffix and  'checkGrad' in args.suffix: 
        record[0,epoch], record[1,epoch], record[2,epoch] = np.mean(grad_block1), np.mean(grad_block2), np.mean(grad_block3)
        grad_block1.clear()
        grad_block2.clear()
        grad_block3.clear()
        np.save(os.path.join(args.root_log, args.store_name, 'record.npy'), record)


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None, mixed=True, autocast=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            with autocast():
                # compute output
                output = model(input)
                output = output[::2] + output[1::2]
                #set_trace()
                loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps, it=0, iters_per_ep=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    elif lr_type == 'ConstantWarmupCosine':
        if epoch<args.warmup:
            lr = args.lr * args.warmup_factor
        else:
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch-args.warmup) / (args.epochs-args.warmup)))
        decay = args.weight_decay
    elif lr_type == 'GradualWarmupCosine':
        if epoch<args.warmup:
            incre_constant = args.lr*(1-args.warmup_factor) / (iters_per_ep*args.warmup)
            lr = args.lr * args.warmup_factor + (it+epoch*iters_per_ep)*incre_constant
        else:
            import math
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch-args.warmup) / (args.epochs-args.warmup)))
        decay = args.weight_decay        #raise NotImplementedError
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    record = np.zeros((3, 50))
    grad_block1 = list()
    grad_block2 = list()
    grad_block3 = list()
    store_block = list()
    main()
