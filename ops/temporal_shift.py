# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class TemporalShift_vit(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False, groups=1, shuffle=False, args=None):
        super(TemporalShift_vit, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.groups = groups
        self.shuffle = shuffle
        self.args = args
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        if self.args.shift_types is not None:
            x = self.various_shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace,
             groups=self.groups, args=self.args)
        else:
            #set_trace()
            x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace,
             groups=self.groups)

        if self.shuffle:
            return channel_shuffle(self.net(x), self.groups)
        else:
            #set_trace()
            #print(x.size())
            return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False, groups=1):
        nt, num_patches, c = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, num_patches, c)
        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :, :fold] = x[:, 1:, :, :fold]  # shift left
        out[:, 1:, :, fold: 2 * fold] = x[:, :-1, :, fold: 2 * fold]  # shift right
        out[:, :, :, 2 * fold:] = x[:, :, :, 2 * fold:]  # not shift
        return out.view(nt, num_patches, c)

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False, groups=1, shuffle=False, args=None):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.groups = groups
        self.shuffle = shuffle
        self.args = args
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        if self.args.shift_types is not None:
            x = self.various_shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace,
             groups=self.groups, args=self.args)
        else:
            #set_trace()
            x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace,
             groups=self.groups)

        if self.shuffle:
            return channel_shuffle(self.net(x), self.groups)
        else:
            #set_trace()
            #print(x.size())
            return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False, groups=1):
        dim_eq_4 = x.dim()==4
        if dim_eq_4:
            # if feat of (bs, c, h, w) shape
            nt, c, h, w = x.size()
            n_batch = nt // n_segment
            x = x.view(n_batch, n_segment, c, h, w)
        else:
            c = x.size(1)
        fold = c // fold_div
        if groups==1 and inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            #raise NotImplementedError  
            out = InplaceShift.apply(x, fold)
        elif groups>1 and inplace:
            out = InplaceGroupShift.apply(x, fold_div)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        if dim_eq_4:
            return out.view(nt, c, h, w)
        else:
            return out



    @staticmethod
    def various_shift(x, n_segment, fold_div=3, inplace=False, groups=1, args=None):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            if groups==1:
                raise ValueError('various_shift is designed for group shift')
            elif groups>1:
                if args.shift_types in ['a', 'b']:
                    out = InplaceGroupShift_ab.apply(x, fold)
                elif 'c' in args.shift_types or 'd' in args.shift_types:
                    times = int(args.shift_types[1:])
                    if 1/fold_div*2*(1+times)>1:
                        raise ValueError
                    out = InplaceGroupShift_cd.apply(x, fold, times)
        else:
            raise NotImplementedError
            if groups==1:
                out = torch.zeros_like(x)
                out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
                out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
                out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
            else:
                raise NotImplementedError

        return out.view(nt, c, h, w)


class TemporalShift_behindARC(TemporalShift):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False, groups=1, shuffle=False, args=None):
        super(TemporalShift_behindARC, self).__init__(net, n_segment=n_segment, n_div=n_div, inplace=inplace, groups=groups, shuffle=shuffle, args=args)

    def forward(self, out, sp_past, sp_now):
        if self.shuffle:
            x = channel_shuffle(self.net(out, sp_past, sp_now), self.groups)
        else:
            #set_trace()
            #print(x.size())
            x = self.net(out, sp_past, sp_now)
        if self.args.shift_types is not None:
            return self.various_shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace,
             groups=self.groups, args=self.args)
        else:
            #set_trace()
            return self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace,
             groups=self.groups)


    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False, groups=1):
        dim_eq_4 = x.dim()==4
        if dim_eq_4:
            # if feat of (bs, c, h, w) shape
            nt, c, h, w = x.size()
            n_batch = nt // n_segment
            x = x.view(n_batch, n_segment, c, h, w)
        else:
            c = x.size(1)
        fold = c // fold_div
        if groups==1 and inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            #raise NotImplementedError
            out = InplaceShift.apply(x, fold)
        elif groups>1 and inplace:
            out = InplaceGroupShift.apply(x, fold_div)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        if dim_eq_4:
            return out.view(nt, c, h, w)
        else:
            return out



class DoubleShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=True, groups=1, shuffle=False):
        super(DoubleShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.groups = groups
        self.shuffle = shuffle
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))
        #import pdb;pdb.set_trace();

    def forward(self, x):
        identity = x
        #import pdb;pdb.set_trace();
        out = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace, groups=self.groups)
        if self.shuffle:
            out = channel_shuffle(out, self.groups)
        out = self.net.conv1(out)
        out = self.net.bn1(out)
        out = self.net.relu(out)

        out = self.shift(out, self.n_segment, fold_div=self.fold_div, inplace=self.inplace, groups=self.groups)
        if self.shuffle:
            out = channel_shuffle(out, self.groups)
        out = self.net.conv2(out)
        out = self.net.bn2(out)

        if self.net.downsample is not None:
            identity = self.net.downsample(x)

        out += identity
        out = self.net.relu(out)
    
        return out

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False, groups=1):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if groups==1 and inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            #raise NotImplementedError
            out = InplaceShift.apply(x, fold)
        elif groups>1 and inplace:
            out = InplaceGroupShift.apply(x, fold_div)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

class InplaceGroupShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        #buffer = input.data.new(n, t, fold, h, w).zero_()
        #buffer[:, :-1] = input.data[:, 1:, 0::fold]
        input.data[:, :-1, 0::fold] = input.data[:, 1:, 0::fold]
        #buffer.zero_()
        #buffer[:, 1:] = input.data[:, :-1, 1:: fold]
        input.data[:, 1:, 1::fold] = input.data[:, :-1, 1::fold]
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        #buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        #buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, 1:, 0::fold] = grad_output.data[:, :-1, 0::fold]
        #buffer.zero_()
        #buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :-1, 1::fold] = grad_output.data[:, 1:, 1::fold] 
        return grad_output, None

'''
class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None
'''

class InplaceGroupShift_ab(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        #buffer = input.data.new(n, t, fold, h, w).zero_()
        #buffer[:, :-1] = input.data[:, 1:, 0::fold]
        input.data[:, :-1, 0:fold] = input.data[:, 1:, 0:fold]
        #buffer.zero_()
        #buffer[:, 1:] = input.data[:, :-1, 1:: fold]
        input.data[:, 1:, fold:2*fold] = input.data[:, :-1, fold:2*fold]
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        #buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        #buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, 1:, 0:fold] = grad_output.data[:, :-1, 0:fold]
        #buffer.zero_()
        #buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :-1, fold:2*fold] = grad_output.data[:, 1:, fold:2*fold] 
        return grad_output, None

class InplaceGroupShift_cd(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold, times):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        ctx.times_ = times
        n, t, c, h, w = input.size()
        #buffer = input.data.new(n, t, fold, h, w).zero_()
        #buffer[:, :-1] = input.data[:, 1:, 0::fold]
        input.data[:, :-1, 0:fold] = input.data[:, 1:, 0:fold]
        #buffer.zero_()
        #buffer[:, 1:] = input.data[:, :-1, 1:: fold]
        input.data[:, 1:, (1+times)*fold:(2+times)*fold] = input.data[:, :-1, (1+times)*fold:(2+times)*fold]
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        times = ctx.times_
        n, t, c, h, w = grad_output.size()
        #buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        #buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, 1:, 0:fold] = grad_output.data[:, :-1, 0:fold]
        #buffer.zero_()
        #buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :-1, (1+times)*fold:(2+times)*fold] = grad_output.data[:, 1:, (1+times)*fold:(2+times)*fold] 
        return grad_output, None, None

class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None


class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False, groups=1, shuffle=False, args=None):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    import torchvision, archs
    #if isinstance(net, (torchvision.models.ResNet, archs.ResNet, archs.halfResNet, archs.CRes_halfResNet, archs.widerResnet)):
    if 'res' in args.arch:
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        #import pdb;pdb.set_trace()
                        #blocks[i] = DoubleShift(b, n_segment=this_segment, n_div=n_div, groups=groups, shuffle=shuffle)
                        blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div,
                         groups=groups, shuffle=shuffle, args=args)
                        if args.suffix and 'ShiftPerConv' in args.suffix and ('18' in args.arch or '34' in args.arch):
                            blocks[i].conv2 = TemporalShift(b.conv2, n_segment=this_segment, n_div=n_div,
                             groups=groups, shuffle=shuffle, args=args)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif place=='naive_skip_connection':
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].identity = TemporalShift(b.identity, n_segment=this_segment, n_div=n_div,
                         groups=groups, shuffle=shuffle, args=args)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
        
        if 'ada' in args.arch and args.suffix and 'ShiftPerConv' in args.suffix:
            #set_trace()
            if (args.where2insert[0]=='1'):
                print('=> Injecting TSM to each step of ARC at stage 1')
                num_blocks = len(net.layer1)
                for b in range(num_blocks):
                    if '18' in args.arch or '34' in args.arch: 
                        net.layer1[b].conv1.net.channel_interacter = TemporalShift_behindARC(net.layer1[b].conv1.net.channel_interacter, n_segment=n_segment, n_div=n_div,
                                                    groups=groups, shuffle=shuffle, args=args)
                        net.layer1[b].conv2.net.channel_interacter = TemporalShift_behindARC(net.layer1[b].conv2.net.channel_interacter, n_segment=n_segment, n_div=n_div,
                                                    groups=groups, shuffle=shuffle, args=args)
                    if '50' in args.arch:
                        net.layer1[b].conv2.net.channel_interacter = TemporalShift_behindARC(net.layer1[b].conv2.net.channel_interacter, n_segment=n_segment, n_div=n_div,
                                                    groups=groups, shuffle=shuffle, args=args)
            if (args.where2insert[1]=='1'):
                print('=> Injecting TSM to each step of ARC at stage 2')
                num_blocks = len(net.layer1)
                num_blocks = len(net.layer2)
                for b in range(num_blocks):
                    if '18' in args.arch or '34' in args.arch:
                        net.layer2[b].conv1.net.channel_interacter = TemporalShift_behindARC(net.layer2[b].conv1.net.channel_interacter, n_segment=n_segment, n_div=n_div,
                                                    groups=groups, shuffle=shuffle, args=args)
                        net.layer2[b].conv2.net.channel_interacter = TemporalShift_behindARC(net.layer2[b].conv2.net.channel_interacter, n_segment=n_segment, n_div=n_div,
                                                    groups=groups, shuffle=shuffle, args=args)
                    if '50' in args.arch:
                        net.layer2[b].conv2.net.channel_interacter = TemporalShift_behindARC(net.layer2[b].conv1.net.channel_interacter, n_segment=n_segment, n_div=n_div,
                                                    groups=groups, shuffle=shuffle, args=args) 
            if (args.where2insert[2]=='1'):
                print('=> Injecting TSM to each step of ARC at stage 3')
                num_blocks = len(net.layer1)
                num_blocks = len(net.layer3)
                for b in range(num_blocks):
                    if '18' in args.arch or '34' in args.arch:
                        net.layer3[b].conv1.net.channel_interacter = TemporalShift_behindARC(net.layer3[b].conv1.net.channel_interacter, n_segment=n_segment, n_div=n_div,
                                                    groups=groups, shuffle=shuffle, args=args)
                        net.layer3[b].conv2.net.channel_interacter = TemporalShift_behindARC(net.layer3[b].conv2.net.channel_interacter, n_segment=n_segment, n_div=n_div,
                                                    groups=groups, shuffle=shuffle, args=args)
                    if '50' in args.arch:
                        net.layer3[b].conv2.net.channel_interacter = TemporalShift_behindARC(net.layer3[b].conv1.net.channel_interacter, n_segment=n_segment, n_div=n_div,
                                                    groups=groups, shuffle=shuffle, args=args)
            if (args.where2insert[3]=='1'):
                print('=> Injecting TSM to each step of ARC at stage 4')
                num_blocks = len(net.layer1)
                num_blocks = len(net.layer4)
                for b in range(num_blocks):
                    if '18' in args.arch or '34' in args.arch:
                        net.layer4[b].conv1.net.channel_interacter = TemporalShift_behindARC(net.layer4[b].conv1.net.channel_interacter, n_segment=n_segment, n_div=n_div,
                                                    groups=groups, shuffle=shuffle, args=args)
                        net.layer4[b].conv2.net.channel_interacter = TemporalShift_behindARC(net.layer4[b].conv2.net.channel_interacter, n_segment=n_segment, n_div=n_div,
                                                    groups=groups, shuffle=shuffle, args=args)
                    if '50' in args.arch:
                        net.layer4[b].conv2.net.channel_interacter = TemporalShift_behindARC(net.layer4[b].conv1.net.channel_interacter, n_segment=n_segment, n_div=n_div,
                                                    groups=groups, shuffle=shuffle, args=args)

    elif 'vit' in args.arch:
        def make_block_temporal(stage, this_segment):
            blocks = list(stage.children())
            print('=> Processing stage with {} blocks'.format(len(blocks)))
            for i, b in enumerate(blocks):
                blocks[i] = TemporalShift_vit(b, n_segment=this_segment, n_div=n_div)
            return nn.Sequential(*(blocks))
        num_blocks = len(net.blocks)
        for i in range(num_blocks):
            net.blocks[i] = TemporalShift_vit(net.blocks[i], n_segment=n_segment, n_div=n_div,
                    groups=groups, shuffle=shuffle, args=args) 

    else:
        raise NotImplementedError(args.arch)


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)
    tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224)
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5

    print('=> Testing GPU...')
    tsm1.cuda()
    tsm2.cuda()
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5
    print('Test passed.')




