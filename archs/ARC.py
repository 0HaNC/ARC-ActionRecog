import torch
import torch.nn as nn
from ipdb import set_trace
import torch.nn.functional as F
import numpy as np
import re
import torch.jit as jit
#torch._C._jit_set_profiling_mode(False)
#torch._C._jit_set_profiling_executor(False)
class Conv2d_bn_relu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, groups=1, bias=False, dilation=1, with_bn=True, no_affine=False, conv_type=nn.Conv2d, norm_type=nn.BatchNorm2d, isReLU=True):
        super(Conv2d_bn_relu, self).__init__()
        self.list = []
        self.list.append(conv_type(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                 padding=padding, groups=groups, bias=bias, dilation=dilation))
        if with_bn:
            self.list.append(norm_type(out_planes, affine=not no_affine))
        if isReLU:
            self.list.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*self.list)
    def forward(self, x):
        return self.block(x)
class Conv3d_bn_relu(Conv2d_bn_relu):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1,
     with_bn=True, no_affine=False, conv_type=nn.Conv3d, norm_type=nn.BatchNorm3d, isReLU=True):
        super(Conv3d_bn_relu, self).__init__(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                 padding=padding, groups=groups, bias=bias, dilation=dilation,
                  with_bn=with_bn, no_affine=False, conv_type=conv_type, norm_type=norm_type, isReLU=isReLU)
class Conv1d_bn_relu(Conv2d_bn_relu):
    def __init__(self):
        super(Conv1d_bn_relu, self).__init__(in_planes, out_planes, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1,
         with_bn=True, no_affine=False, conv_type=nn.Conv3d, norm_type=nn.BatchNorm3d, isReLU=True)

@torch.jit.script
def affine(x,m,a):
    return x*m+a
#  two warmup iterations with 1.7+ JIT (profiling & codegen)
 
class BN_affine(nn.Module):
    def __init__(self, in_planes, w_init, b_init):
        super(BN_affine, self).__init__()
        self.w = nn.Parameter(torch.ones(in_planes,1,1) * w_init)
        self.b = nn.Parameter(torch.ones(in_planes,1,1) * b_init)
        self.gate = nn.Identity()
    #@jit.script
    def _affine(x,m,a):
        return x*m+a
    def forward(self, x):
        #N, C = x.size(0), self.w.size(0)
        #self.ex_w = self.w.expand(N, C).contiguous().view(N, C, 1, 1)
        #self.ex_b = self.b.expand(N, C).contiguous().view(N, C, 1, 1)
        #return self.w*x
        #set_trace()
        #return _affine(x, self.ex_w, self.ex_b)
        #return x*self.w + self.b
        return affine(x, self.w, self.b)

class ARC_layer(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3,
                     padding=0, groups=1, bias=False, dilation=1, args=None, layer_info=None, isReLU=True):
        super(ARC_layer, self).__init__()
        self.args = args
        self.interacter, self.with_bn, self.insert_propotion, self.t =\
         args.interacter, args.with_bn, args.insert_propotion, args.num_segments
        if layer_info:
            self.layer_info = layer_info
        elif args.layer_info:
            self.layer_info = args.layer_info
        else:
            raise ValueError('Both layer_info and args.layer_info are None!')
 
        if 'SepBN' in self.interacter or 'reparam' in self.interacter:
            self.hook_store = [None]
            def forward_store_hook(module, input, output):
                #set_trace()
                self.hook_store[0] = output
            self.hook_fun = forward_store_hook
        self.no_affine = 'SepJitBN' in self.interacter
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        if self.insert_propotion=='all' and self.stride != 1:
            #self.upsample = nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=False)
            # self.upsample = nn.Upsample(size=(2,2), mode='nearest')
            self.upsample = nn.Upsample(scale_factor=(2,2), mode='nearest')
        self.out_folds = []
        #propotions = [int(re.search(r'\d+', i).group()) for i in self.layer_info]
        #set_trace()
        propotions = [int(i[-1]) for i in self.layer_info]
        for i in self.layer_info:
            self.out_folds.append(int(out_planes * int(i[-1])/sum(propotions)))
        self.out_folds = self.out_folds
        # align channels with the orignal one
        self.out_folds[-1] = out_planes - sum(self.out_folds[:-1])
        class full_shifter(nn.Module):
            def __init__(self):
                super(full_shifter, self).__init__()
            def _channel_shift(self, cur_out, sp_now):
                num_to_shift = cur_out.size(1)
                return torch.cat([cur_out, sp_now[:, num_to_shift:]], 1)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_shifter(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_shifter, self).__init__()
                self.out_folds = Mconver.out_folds[:-1]
                num_weights_list = torch.tensor(self.out_folds)
                num_weights_list = num_weights_list.cumsum(dim=0)
                # For cross-stage compatibility
                self.in_planes = Mconver.in_planes
                num_weights_list[num_weights_list>self.in_planes] = self.in_planes
                #print(num_weights_list)
                num_weights = num_weights_list.sum()
                weight = torch.zeros(num_weights, 1, 1)
                #weight = -5 * torch.ones(num_weights)
                #weight = weight.cuda()
                self.Weight = nn.Parameter(weight)
                self.start_ind = 0
                self.sigm = nn.Sigmoid()
                self.T = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            def _channel_shift(self, cur_out, sp_now):
                num_to_shift = cur_out.size(1)
                num_to_shift = num_to_shift if num_to_shift<=self.in_planes else self.in_planes
                raw_weights = self.Weight[self.start_ind: self.start_ind+num_to_shift]
                #weights_as_prob = self.sigm(raw_weights / self.T).unsqueeze(dim=-1). unsqueeze(dim=-1)
                weights_as_prob = self.sigm(raw_weights / self.T)
                #interacted_feat = cur_out.detach()*weights_as_prob + sp_now[:, :num_to_shift]*(1-weights_as_prob)
                interacted_feat = sp_now.clone()
                interacted_feat[:, :num_to_shift] = cur_out[:, :num_to_shift]*(weights_as_prob) + sp_now[:, :num_to_shift]*(1-weights_as_prob)
                #interacted_feat = cur_out*weights_as_prob + sp_now[:, :num_to_shift]*(1-weights_as_prob)
                #interacted_feat = cur_out*weights_as_prob + sp_now[:, :num_to_shift]
                self.start_ind += num_to_shift
                if self.start_ind==self.Weight.size(0):
                    self.start_ind = 0
                #return torch.cat([interacted_feat, sp_now[:, num_to_shift:]], 1)
                return interacted_feat
            def forward(self, out, sp_past, sp_now):
                if torch.tensor(self.out_folds).sum() > sp_now.size(1):
                    #raise ValueError('num of out channels should not be greater than the one of sp_now.')
                    pass
                return self._channel_shift(out, sp_now)
        class ada_full_shifter_babies(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_shifter_babies, self).__init__()
                self.out_folds = Mconver.out_folds[:-1]
                num_weights_list = torch.tensor(self.out_folds).cumsum(dim=0)
                #print(num_weights_list)
                num_weights = num_weights_list.sum()
                weight = -5 * torch.ones(num_weights, 2)
                weight = weight.cuda()
                self.Weight = nn.Parameter(weight)
                self.start_ind = 0
                self.sigm = nn.Sigmoid()
                self.T = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            def _channel_shift(self, cur_out, sp_now):
                num_to_shift = cur_out.size(1)
                #print(self.start_ind, num_to_shift, self.Weight.size())
                raw_weights = self.Weight[self.start_ind: self.start_ind+num_to_shift, 0]
                weights_as_prob = self.sigm(raw_weights / self.T).unsqueeze(dim=-1). unsqueeze(dim=-1)
                raw_weights2 = self.Weight[self.start_ind: self.start_ind+num_to_shift, 1]
                weights_as_prob2 = self.sigm(raw_weights2 / self.T).unsqueeze(dim=-1). unsqueeze(dim=-1)
                #interacted_feat = cur_out.detach()*weights_as_prob + sp_now[:, :num_to_shift]*(1-weights_as_prob)
                #interacted_feat = cur_out*weights_as_prob + sp_now[:, :num_to_shift]*(1-weights_as_prob)
                try:
                    interacted_feat = cur_out*weights_as_prob + sp_now[:, :num_to_shift]*weights_as_prob2
                except:
                    set_trace()
                self.start_ind += num_to_shift
                if self.start_ind==self.Weight.size(0):
                    self.start_ind = 0
                return torch.cat([interacted_feat, sp_now[:, num_to_shift:]], 1)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full(nn.Module):
            def __init__(self, Mconver):
                super(ada_full, self).__init__()
                self.out_folds = Mconver.out_folds[:-1]
                num_weights_list = torch.tensor(self.out_folds).cumsum(dim=0)
                #print(num_weights_list)
                num_weights = num_weights_list.sum()
                weight = torch.zeros(num_weights, 2)
                weight = weight.cuda()
                self.Weight = nn.Parameter(weight)
                self.start_ind = 0
                #self.sigm = nn.Sigmoid()
                self.sigm = nn.Identity()
                self.T = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            def _channel_shift(self, cur_out, sp_now):
                num_to_shift = cur_out.size(1)
                #print(self.start_ind, num_to_shift, self.Weight.size())
                raw_weights = self.Weight[self.start_ind: self.start_ind+num_to_shift, 0]
                weights_as_prob = self.sigm(raw_weights / self.T).unsqueeze(dim=-1). unsqueeze(dim=-1)
                raw_weights2 = self.Weight[self.start_ind: self.start_ind+num_to_shift, 1]
                weights_as_prob2 = self.sigm(raw_weights2 / self.T).unsqueeze(dim=-1). unsqueeze(dim=-1)
                #interacted_feat = cur_out.detach()*weights_as_prob + sp_now[:, :num_to_shift]*(1-weights_as_prob)
                interacted_feat = cur_out*weights_as_prob + sp_now[:, :num_to_shift]*weights_as_prob2
                self.start_ind += num_to_shift
                if self.start_ind==self.Weight.size(0):
                    self.start_ind = 0
                return torch.cat([interacted_feat, sp_now[:, num_to_shift:]], 1)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction, self).__init__()
                self.out_folds = Mconver.out_folds[:-1]
                '''
                num_weights_list = torch.tensor(self.out_folds).cumsum(dim=0)
                num_weights = num_weights_list.sum()
                weight = torch.zeros(num_weights, 2)
                weight = weight.cuda()
                self.Weight = nn.Parameter(weight)
                self.start_ind = 0
                self.sigm = nn.Identity()
                self.T = nn.Parameter(torch.tensor(1.0), requires_grad=False)
                '''
                self.planes = torch.tensor(self.out_folds).sum()
                # Weight_mat of shape(out_channels, in_channels/groups, kH, kW)
                #Weight_mat = torch.randn(self.planes, self.planes, 1, 1).cuda()
                Weight_mat = torch.zeros(self.planes, self.planes, 1, 1).cuda()
                self.Weight_mat = nn.Parameter(Weight_mat)
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                num_to_shift = cur_out.size(1)
                #interacted_feat = cur_out.detach()*weights_as_prob + sp_now[:, :num_to_shift]*(1-weights_as_prob)
                raw_weights = self.Weight_mat[:num_to_shift, :num_to_shift]
                interacted_feat = F.conv2d(cur_out, raw_weights) + sp_now[:, :num_to_shift]
                interacted_feat = self.relu(interacted_feat)
                return torch.cat([interacted_feat, sp_now[:, num_to_shift:]], 1)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction1_V2(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2, self).__init__()
                self.start_ind = 0
                self.out_folds = Mconver.out_folds[:-1]
                #self.out_planes = Mconver.out_planes
                self.in_planes = Mconver.in_planes
                #self.planes = torch.tensor(self.out_folds).sum()
                self.planes = sum(self.out_folds)
                # (out_chan, in_chan/groups, kH, kW)
                Weight_mat = torch.zeros(self.in_planes,self.planes).unsqueeze(-1).unsqueeze(-1)
                self.Weight_mat = nn.Parameter(Weight_mat)
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    self.cache = self.cache + F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache = self.cache + F.conv3d(cur_out[:, self.start_ind:num_to_shift], raw_weights.unsqueeze(2))
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                return self.relu(self.cache)
                #return self.cache
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction1_V2_SepPoolPW(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_SepPoolPW, self).__init__()
                self.out_folds = Mconver.out_folds[:-1]
                self.in_planes = Mconver.in_planes
                self.planes = sum(self.out_folds)
                self.cumsumed_out_folds = torch.tensor([0]+self.out_folds[:-1]).cumsum(dim=0)
                Weight_mat = torch.zeros(self.in_planes, self.cumsumed_out_folds.sum()).unsqueeze(-1).unsqueeze(-1)
                self.Weight_mat = nn.Parameter(Weight_mat)
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, :num_to_shift]
                if sp_now.dim()==4:
                    self.cache =  sp_now + F.conv2d(cur_out.mean((-1,-2),keepdim=True), raw_weights)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache = sp_now + self.cache + F.conv3d(cur_out[:, num_to_shift], raw_weights.unsqueeze(2))
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                return self.relu(self.cache)
                #return self.cache
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction1_V2_RNNPoolPW(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_RNNPoolPW, self).__init__()
                self.out_folds = Mconver.out_folds[:-1]
                self.in_planes = Mconver.in_planes
                self.planes = sum(self.out_folds)
                self.rnn = nn.GRUCell(input_size=self.planes, hidden_size=self.in_planes)
                self.cumsumed_out_folds = torch.tensor([0]+self.out_folds[:-1]).cumsum(dim=0)
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
                self.start_ind = 0
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.hidden = torch.zeros(sp_now.size()[:2]).to(sp_now.device)
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                if sp_now.dim()==4:
                    rnn_input = torch.zeros(sp_now.size(0), self.planes).to(sp_now.device)
                    rnn_input[:, :cur_out.size(1)] = cur_out.mean((-1,-2),keepdim=False)
                    self.hidden = self.rnn(rnn_input, self.hidden)
                    #set_trace()
                    self.cache =  sp_now + self.hidden.unsqueeze(-1).unsqueeze(-1)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache = sp_now + self.cache + F.conv3d(cur_out[:, num_to_shift], raw_weights.unsqueeze(2))
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                return self.relu(self.cache)
                #return self.cache
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)


        class ada_full_PWInteraction1_V2_dwConv(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_dwConv, self).__init__(Mconver)
                self.dw_weights = torch.zeros(self.in_planes, 1, 3, 3)
                self.dw_weights = nn.Parameter(dw_weights)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    self.cache = self.cache + F.conv2d(F.conv2d(cur_out[:, self.start_ind:num_to_shift],
                     raw_weights),
                     dw_weights, groups=self.in_planes)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache = self.cache + F.conv3d(F.conv3d(cur_out[:, self.start_ind:num_to_shift], 
                    	raw_weights.unsqueeze(2)),
                    	dw_weights.unsqueeze(2), groups=self.in_planes)
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                return self.relu(self.cache)
        class ada_full_PWInteraction1_V2_dwConv2(ada_full_PWInteraction1_V2):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_dwConv2, self).__init__(Mconver)
                dw_weights = torch.zeros(self.planes, 1, 3, 3)
                self.dw_weights = nn.Parameter(dw_weights)
                nn.init.constant_(self.dw_weights, 1e-6)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                raw_weights_dw = self.dw_weights[self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    try:
                        self.cache = self.cache + F.conv2d(F.conv2d(cur_out[:, self.start_ind:num_to_shift],
                         raw_weights_dw, groups=num_to_shift-self.start_ind, padding=1),
                         raw_weights)
                    except:
                        set_trace()
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache = self.cache + F.conv3d(F.conv3d(cur_out[:, self.start_ind:num_to_shift], 
                        raw_weights_dw.unsqueeze(2), groups=num_to_shift-self.start_ind, padding=1),
                        raw_weights.unsqueeze(2))
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                return self.relu(self.cache)
        class ada_full_PWInteraction1_V2_dwConv2BN(ada_full_PWInteraction1_V2):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_dwConv2BN, self).__init__(Mconver)
                dw_weights = torch.zeros(self.planes, 1, 3, 3)
                self.dw_weights = nn.Parameter(dw_weights)
                nn.init.constant_(self.dw_weights, 1e-6)
                #set bns
                self.bns = []
                self.c2i = {}
                c_cum = 0
                for i, c in enumerate(self.out_folds):
                    self.c2i[c_cum] = i
                    c_cum += c
                    self.bns.append(nn.BatchNorm2d(c))
                self.bns = nn.Sequential(*self.bns)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                raw_weights_dw = self.dw_weights[self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    try:
                        self.cache = self.cache + F.conv2d(
                            self.relu(self.bns[self.c2i[self.start_ind]](
                                F.conv2d(cur_out[:, self.start_ind:num_to_shift],
                         raw_weights_dw, groups=num_to_shift-self.start_ind, padding=1))),
                         raw_weights)
                    except:
                        set_trace()
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache = self.cache + F.conv3d(F.conv3d(cur_out[:, self.start_ind:num_to_shift], 
                        raw_weights_dw.unsqueeze(2), groups=num_to_shift-self.start_ind, padding=1),
                        raw_weights.unsqueeze(2))
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                return self.relu(self.cache)
        class ada_full_PWInteraction1_V2_pool(ada_full_PWInteraction1_V2):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_pool, self).__init__(Mconver)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]

                if sp_now.dim()==4:
                    self.cache = self.cache + F.conv2d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache = self.cache + F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights.unsqueeze(2))
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                return self.relu(self.cache)
        class ada_full_PWInteraction1_V2_pool_CAttend(ada_full_PWInteraction1_V2):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_pool_CAttend, self).__init__(Mconver)
                self.gate = nn.Sigmoid()
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = 0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    self.cache = self.cache + F.conv2d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache = self.cache + F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights.unsqueeze(2))
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                return self.gate(self.cache)*sp_now
        class ada_full_PWInteraction1_V2_pool_CAttend_add(ada_full_PWInteraction1_V2):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_pool_CAttend_add, self).__init__(Mconver)
                self.gate = nn.Sigmoid()
                Weight_mat_add = torch.zeros(self.in_planes,self.planes).unsqueeze(-1).unsqueeze(-1)
                self.Weight_mat_add = nn.Parameter(Weight_mat_add)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                raw_weights_add = self.Weight_mat_add[:, self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    self.cache_attend = self.cache_attend + F.conv2d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights)
                    self.cache_add = self.cache_add + F.conv2d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights_add)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache_attend = self.cache_attend + F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights.unsqueeze(2))
                    self.cache_add = self.cache_add + F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights_add.unsqueeze(2))                    
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                return self.gate(self.cache_attend)*sp_now + self.cache_add
        class ada_full_PWInteraction1_V2_pool_CAttend_add_woCache(ada_full_PWInteraction1_V2_pool_CAttend_add):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_pool_CAttend_add_woCache, self).__init__(Mconver)
            def _channel_shift(self, cur_out, sp_now):
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                raw_weights_add = self.Weight_mat_add[:, self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    self.cache_attend =  F.conv2d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights)
                    self.cache_add = F.conv2d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights_add)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache_attend = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights.unsqueeze(2))
                    self.cache_add = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights_add.unsqueeze(2))                    
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                return self.gate(self.cache_attend)*sp_now + self.cache_add
        class ada_full_PWInteraction1_V2_pool_CAttend_add_KeepSpatial(ada_full_PWInteraction1_V2_pool_CAttend_add):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_pool_CAttend_add_KeepSpatial, self).__init__(Mconver)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                raw_weights_add = self.Weight_mat_add[:, self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    self.cache_attend = self.cache_attend +  F.conv2d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights)
                    self.cache_add = self.cache_add + F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights_add)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache_attend = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights.unsqueeze(2))
                    self.cache_add = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights_add.unsqueeze(2))                    
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                return self.relu(self.gate(self.cache_attend)*2*sp_now + self.cache_add)
        class ada_full_PWInteraction1_V2_pool_CAttendByAdd_KeepSpatial(ada_full_PWInteraction1_V2_pool_CAttend_add):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_pool_CAttendByAdd_KeepSpatial, self).__init__(Mconver)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attendByAdd = sp_now
                    self.cache_add = 0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                raw_weights_add = self.Weight_mat_add[:, self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    self.cache_attendByAdd = self.cache_attendByAdd +  F.conv2d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights)
                    self.cache_add = self.cache_add + F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights_add)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache_attend = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights.unsqueeze(2))
                    self.cache_add = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights_add.unsqueeze(2))                    
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                # Modeling Channel attention via Addition and ReLU
                return self.relu(self.relu(self.cache_attendByAdd) + self.cache_add)
        class ada_full_PWInteraction1_V2_pool_CAttendByAdd_KeepSpatialDWConv(ada_full_PWInteraction1_V2_pool_CAttend_add):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_pool_CAttendByAdd_KeepSpatialDWConv, self).__init__(Mconver)
                #set dw_weights
                dw_weights = torch.zeros(self.planes, 1, 3, 3)
                self.dw_weights = nn.Parameter(dw_weights)
                nn.init.constant_(self.dw_weights, 1e-6)
                #set bns
                self.bns = []
                self.c2i = {}
                c_cum = 0
                for i, c in enumerate(self.out_folds):
                    self.c2i[c_cum] = i
                    c_cum += c
                    self.bns.append(nn.BatchNorm2d(c))
                self.bns = nn.Sequential(*self.bns)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attendByAdd = sp_now
                    self.cache_add = 0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                raw_weights_add = self.Weight_mat_add[:, self.start_ind:num_to_shift]
                raw_weights_dw = self.dw_weights[self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    sp_past = cur_out[:, self.start_ind:num_to_shift]
                    self.cache_attendByAdd = self.cache_attendByAdd +  F.conv2d(sp_past.mean((-1,-2),keepdim=True), raw_weights_add)
                    self.cache_add = self.cache_add + F.conv2d(
                        self.relu(
                        self.bns[self.c2i[self.start_ind]](
                        F.conv2d(sp_past,
                     raw_weights_dw, groups=num_to_shift-self.start_ind, padding=1))),
                     raw_weights)
                    ##################???????????????????????????????????????????????????????????????? forget to comment it?
                    #self.cache_add = self.cache_add + F.conv2d(sp_past, raw_weights_add)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache_attend = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights.unsqueeze(2))
                    self.cache_add = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights_add.unsqueeze(2))                    
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                # Modeling Channel attention via Addition and ReLU
                return self.relu(self.relu(self.cache_attendByAdd) + self.cache_add)
        class ada_full_PWInteraction1_V2_STpool_CAttendByAdd_KeepSpatialDWConv(ada_full_PWInteraction1_V2_pool_CAttend_add):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_STpool_CAttendByAdd_KeepSpatialDWConv, self).__init__(Mconver)
                self.num_segments = Mconver.args.num_segments
                #set dw_weights
                dw_weights = torch.zeros(self.planes, 1, 3, 3)
                self.dw_weights = nn.Parameter(dw_weights)
                self.Weight_matT = nn.Parameter(torch.zeros(self.in_planes, self.planes, 1, 1))
                nn.init.constant_(self.dw_weights, 1e-6)
                #set bns
                self.bns = []
                self.c2i = {}
                c_cum = 0
                for i, c in enumerate(self.out_folds):
                    self.c2i[c_cum] = i
                    c_cum += c
                    self.bns.append(nn.BatchNorm2d(c))
                self.bns = nn.Sequential(*self.bns)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attendByAdd = sp_now
                    self.cache_add = 0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                raw_weights_add = self.Weight_mat_add[:, self.start_ind:num_to_shift]
                raw_weightsT = self.Weight_matT[:, self.start_ind:num_to_shift]
                raw_weights_dw = self.dw_weights[self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    sp_past = cur_out[:, self.start_ind:num_to_shift]
                    sp_past_TPool_stView = sp_past.view((-1, self.num_segments) + (sp_past.size()[-3:])).mean(1)
                    self.cache_attendByAdd = self.cache_attendByAdd + \
                     F.conv2d(sp_past.mean((-1,-2),keepdim=True), raw_weights_add) + \
                     F.conv2d(sp_past_TPool_stView, raw_weightsT).repeat_interleave(self.num_segments, dim=0)
                    self.cache_add = self.cache_add + F.conv2d(
                        self.relu(
                        self.bns[self.c2i[self.start_ind]](
                        F.conv2d(sp_past,
                     raw_weights_dw, groups=num_to_shift-self.start_ind, padding=1))),
                     raw_weights)
                    ##################???????????????????????????????????????????????????????????????? forget to comment it?
                    #self.cache_add = self.cache_add + F.conv2d(sp_past, raw_weights_add)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache_attend = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights.unsqueeze(2))
                    self.cache_add = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights_add.unsqueeze(2))                    
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                # Modeling Channel attention via Addition and ReLU
                return self.relu(self.relu(self.cache_attendByAdd) + self.cache_add)

        class ada_full_PWInteraction1_V2_STpool_CAttendByAdd_KeepSpatialDWConv_PartShare(ada_full_PWInteraction1_V2_pool_CAttend_add):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_STpool_CAttendByAdd_KeepSpatialDWConv_PartShare, self).__init__(Mconver)
                self.num_segments = Mconver.args.num_segments
                self.reduce_attend = Mconver.args.reduce_attend
                del self.Weight_mat_add
                #set dw_weights
                self.dw_weights = nn.Parameter(torch.zeros(self.planes, 1, 3, 3))
                self.Weight_matT = nn.Parameter(torch.zeros(self.planes//self.reduce_attend, self.planes, 1, 1))
                self.Weight_matS = nn.Parameter(torch.zeros(self.planes//self.reduce_attend, self.planes, 1, 1))
                self.dw_weightsT = nn.Parameter(torch.zeros(self.planes, 1, 3, 3))
                self.Weight_matShared = nn.Parameter(torch.zeros(self.in_planes, self.planes//self.reduce_attend, 1, 1))
                nn.init.kaiming_normal_(self.Weight_matT, mode='fan_in')
                nn.init.kaiming_normal_(self.Weight_matS, mode='fan_in')
                nn.init.kaiming_normal_(self.dw_weights, mode='fan_in')
                nn.init.kaiming_normal_(self.dw_weightsT, mode='fan_in')
                #set bns
                self.bns = []
                self.bnsT = []
                self.c2i = {}
                c_cum = 0
                for i, c in enumerate(self.out_folds):
                    self.c2i[c_cum] = i
                    c_cum += c
                    self.bns.append(nn.BatchNorm2d(c))
                self.bns = nn.Sequential(*self.bns)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attendByAdd = sp_now
                    self.cache_add = 0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                raw_weightsS = self.Weight_matS[:, self.start_ind:num_to_shift]
                raw_weightsT = self.Weight_matT[:, self.start_ind:num_to_shift]
                raw_weights_dw = self.dw_weights[self.start_ind:num_to_shift]
                raw_weights_dwT = self.dw_weightsT[self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    sp_past = cur_out[:, self.start_ind:num_to_shift]
                    sp_past_TPool_stView = sp_past.view((-1, self.num_segments) + (sp_past.size()[-3:])).mean(1)
                    self.cache_attendByAdd = self.cache_attendByAdd + \
                     F.conv2d(F.conv2d(sp_past.mean((-1,-2),keepdim=True), raw_weightsS), self.Weight_matShared) + \
                     F.conv2d(F.conv2d(F.conv2d(sp_past_TPool_stView, raw_weightsT)
                     	, raw_weights_dwT, groups=num_to_shift-self.start_ind, padding=1)
                     	, self.Weight_matShared).repeat_interleave(self.num_segments, dim=0)
                    self.cache_add = self.cache_add + F.conv2d(
                        self.relu(
                        self.bns[self.c2i[self.start_ind]](
                        F.conv2d(sp_past,
                     raw_weights_dw, groups=num_to_shift-self.start_ind, padding=1))),
                     raw_weights)
                    ##################???????????????????????????????????????????????????????????????? forget to comment it?
                    #self.cache_add = self.cache_add + F.conv2d(sp_past, raw_weights_add)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache_attend = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights.unsqueeze(2))
                    self.cache_add = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights_add.unsqueeze(2))                    
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                # Modeling Channel attention via Addition and ReLU
                return self.relu(self.relu(self.cache_attendByAdd) + self.cache_add)

        class ada_full_PWInteraction1_V2_pool_CAttendByAdd_KeepSpatialDWConv_FLOPs(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_pool_CAttendByAdd_KeepSpatialDWConv_FLOPs, self).__init__()
                self.out_folds = Mconver.out_folds[:-1]
                self.in_planes = Mconver.in_planes
                self.start_ind = 0
                self.bns = []
                self.dws = []
                self.denses_attend = []
                self.denses_add = []
                self.c2i = {}
                c_cum = 0
                for i, c in enumerate(self.out_folds):
                    self.c2i[c_cum] = i
                    c_cum += c
                    self.bns.append(nn.BatchNorm2d(c))
                    self.dws.append(nn.Conv2d(c, c, kernel_size=3, groups=c, padding=1, bias=False))
                    self.denses_attend.append(nn.Conv2d(c, self.in_planes, kernel_size=1, groups=1, bias=False))
                    self.denses_add.append(nn.Conv2d(c, self.in_planes, kernel_size=1, groups=1, bias=False))
                self.bns = nn.Sequential(*self.bns)
                self.dws = nn.Sequential(*self.dws)
                self.denses_attend = nn.Sequential(*self.denses_attend)
                self.denses_add = nn.Sequential(*self.denses_add)
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attendByAdd = sp_now
                    self.cache_add = 0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                if sp_now.dim()==4:
                    self.cache_attendByAdd = self.cache_attendByAdd +  self.denses_attend[self.c2i[self.start_ind]](
                        cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True))
                    self.cache_add = self.cache_add + self.denses_add[self.c2i[self.start_ind]](
                        self.relu(
                        self.bns[self.c2i[self.start_ind]](
                        self.dws[self.c2i[self.start_ind]](cur_out[:, self.start_ind:num_to_shift])
                        )))
                    #self.cache_add = self.cache_add + F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights_add)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache_attend = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights.unsqueeze(2))
                    self.cache_add = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights_add.unsqueeze(2))                    
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                #set_trace()
                # Modeling Channel attention via Addition and ReLU
                return self.relu(self.relu(self.cache_attendByAdd) + self.cache_add)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)


        class ada_full_PWInteraction1_V2_pool_CAttendByAdd_KeepSpatial_ForceDiverse(ada_full_PWInteraction1_V2_pool_CAttend_add):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_pool_CAttendByAdd_KeepSpatial_ForceDiverse, self).__init__(Mconver)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attendByAdd = sp_now
                    self.cache_add = 0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                raw_weights_add = self.Weight_mat_add[:, self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    self.cache_attendByAdd = self.cache_attendByAdd +  F.conv2d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights)
                    self.cache_add = self.cache_add + F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights_add)
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache_attend = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights.unsqueeze(2))
                    self.cache_add = F.conv3d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights_add.unsqueeze(2))                    
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                # Modeling Channel attention via Addition and ReLU
                attended = self.relu(self.cache_attendByAdd)
                # detach is important
                signs = (self.relu(-self.cache_attendByAdd.detach())).sign()
                return self.relu(attended + (-signs)*(self.cache_add.abs()))
                return self.relu(attended + (-signs)*self.cache_add)

        class ada_full_PWInteraction1_V2_pool_CAttend_SharedDWConvadd(ada_full_PWInteraction1_V2):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_pool_CAttend_SharedDWConvadd, self).__init__(Mconver)
                self.gate = nn.Sigmoid()
                self.temporature = .001
                dw_weights = torch.zeros(self.in_planes, 1, 3, 3)
                self.dw_weights = nn.Parameter(dw_weights)
                nn.init.kaiming_normal_(self.dw_weights)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    self.cache = F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights)
                    self.cache_attend = self.cache_attend + self.cache.mean((-1,-2), keepdim=True)
                    self.cache_add = self.cache_add + F.conv2d(self.cache, self.dw_weights, groups=self.in_planes, padding=1)
                    #self.cache_add = self.cache_add + self.cache
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache = F.conv3d(cur_out[:, self.start_ind:num_to_shift], raw_weights.unsqueeze(2))
                    self.cache_attend = self.cache_attend + self.cache.mean((-1,-2), keepdim=True)
                    self.cache_add = self.cache_add + F.conv3d(self.cache, self.dw_weights.unsqueeze(2), groups=self.in_planes, padding=1)
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                #return self.relu(self.cache_add)
                #return self.gate(self.cache_attend/self.temporature)*sp_now + self.relu(self.cache_add)
                return self.gate(self.cache_attend/self.temporature)*sp_now + (1-self.gate(self.cache_attend/self.temporature))*self.relu(self.cache_add)
        class ada_full_PWInteraction1_V2_pool_CAttend_SharedDWConvadd2(ada_full_PWInteraction1_V2):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_pool_CAttend_SharedDWConvadd2, self).__init__(Mconver)
                self.gate = nn.Sigmoid()
                self.temporature = .001
                dw_weights = torch.zeros(self.planes, 1, 3, 3)
                self.dw_weights = nn.Parameter(dw_weights)
                nn.init.kaiming_normal_(self.dw_weights)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                raw_weights_dw = self.dw_weights[self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    self.cache_attend = self.cache_attend + F.conv2d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights)
                    self.cache_add = self.cache_add + F.conv2d(F.conv2d(cur_out[:, self.start_ind:num_to_shift], 
                    raw_weights_dw, groups=num_to_shift-self.start_ind, padding=1),
                    raw_weights)
                    #self.cache_add = self.cache_add + self.cache
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache_attend = self.cache_attend + self.cache.mean((-1,-2), keepdim=True)
                    self.cache_add = self.cache_add + F.conv3d(self.cache, self.dw_weights.unsqueeze(2), groups=self.in_planes, padding=1)
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                return self.gate(self.cache_attend/self.temporature)*sp_now + (1-self.gate(self.cache_attend/self.temporature))*self.relu(self.cache_add)
        class ada_full_PWInteraction1_V2_pool_CAttend_SharedDWConvadd2_bn(ada_full_PWInteraction1_V2):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_pool_CAttend_SharedDWConvadd2_bn, self).__init__(Mconver)
                self.gate = nn.Sigmoid()
                self.temporature = .001
                dw_weights = torch.zeros(self.planes, 1, 3, 3)
                self.dw_weights = nn.Parameter(dw_weights)
                nn.init.kaiming_normal_(self.dw_weights)
                #set bns
                self.bns = []
                self.c2i = {}
                c_cum = 0
                for i, c in enumerate(self.out_folds):
                    self.c2i[c_cum] = i
                    c_cum += c
                    self.bns.append(nn.BatchNorm2d(c))
                self.bns = nn.Sequential(*self.bns)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                raw_weights_dw = self.dw_weights[self.start_ind:num_to_shift]
                if sp_now.dim()==4:
                    self.cache_attend = self.cache_attend + F.conv2d(cur_out[:, self.start_ind:num_to_shift].mean((-1,-2),keepdim=True), raw_weights)
                    self.cache_add = self.cache_add + F.conv2d(self.bns[self.c2i[self.start_ind]](F.conv2d(cur_out[:, self.start_ind:num_to_shift], 
                    raw_weights_dw, groups=num_to_shift-self.start_ind, padding=1)),
                    raw_weights)
                    #self.cache_add = self.cache_add + self.cache
                # To be compatible with SlowOnly
                elif sp_now.dim()==5:
                    self.cache_attend = self.cache_attend + self.cache.mean((-1,-2), keepdim=True)
                    self.cache_add = self.cache_add + F.conv3d(self.cache, self.dw_weights.unsqueeze(2), groups=self.in_planes, padding=1)
                else:
                    raise NotImplementedError('len(sp_now)=={}'.format(len(sp_now)))
                self.start_ind = num_to_shift
                return self.gate(self.cache_attend/self.temporature)*sp_now + (1-self.gate(self.cache_attend/self.temporature))*self.relu(self.cache_add)

        class ada_full_PWInteraction1_V2_woReLU(ada_full_PWInteraction1_V2):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_woReLU, self).__init__(Mconver)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                self.cache = self.cache + F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights)
                self.start_ind = num_to_shift
                return self.cache
        class ada_full_PWInteraction1_V2_bias(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_bias, self).__init__()
                self.start_ind = 0
                self.out_folds = Mconver.out_folds[:-1]
                #self.out_planes = Mconver.out_planes
                self.in_planes = Mconver.in_planes
                #self.planes = torch.tensor(self.out_folds).sum()
                self.planes = sum(self.out_folds)
                self.denses = []
                for i in range(len(self.out_folds)):
                    self.denses.append(nn.Conv2d(self.out_folds[i], self.in_planes, kernel_size=1))
                self.denses = nn.Sequential(*self.denses)
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                #self.cache = self.cache + F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights) + self.bias[:, self.start_ind//self.out_folds[0]]
                self.cache = self.cache + self.denses[self.start_ind//self.out_folds[0]](cur_out[:, self.start_ind:num_to_shift])
                self.start_ind = num_to_shift
                return self.relu(self.cache)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction1_V2_MLP2(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_MLP2, self).__init__()
                self.start_ind = 0
                self.out_folds = Mconver.out_folds[:-1]
                #self.out_planes = Mconver.out_planes
                self.in_planes = Mconver.in_planes
                #self.planes = torch.tensor(self.out_folds).sum()
                self.planes = sum(self.out_folds)
                Weight_mat = 1e-4*torch.ones(int(self.planes/Mconver.args.reduce), self.planes).unsqueeze(-1).unsqueeze(-1)
                #Weight_mat = torch.zeros(int(self.planes/Mconver.args.reduce), self.planes).unsqueeze(-1).unsqueeze(-1)
                self.Weight_mat = nn.Parameter(Weight_mat)
                self.dense2 = nn.Conv2d(int(self.planes/Mconver.args.reduce), self.in_planes, kernel_size=1,bias=False)
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                self.cache = self.cache + self.dense2(self.relu(F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights)))
                self.start_ind = num_to_shift
                return self.relu(self.cache)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction1_V2_FLOPs(nn.Module):
            # mimic FLOPs of PWI1
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_FLOPs, self).__init__()
                self.start_ind = 0
                self.out_folds = Mconver.out_folds[:-1]
                self.in_planes = Mconver.in_planes
                self.planes = sum(self.out_folds)
                self.dense = nn.Conv2d(self.planes, self.in_planes, kernel_size=1, bias=False)
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                    # mimic FLOPs of PWI1
                    bt, c, h, w = sp_now.size()
                    self.dense_calc = self.dense(torch.zeros(bt, self.planes, h, w))
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                # mimic FLOPs of PWI1
                self.cache = self.cache + self.dense_calc
                self.start_ind = num_to_shift
                return self.relu(self.cache)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        '''
        class ada_full_PWInteraction1_V2_bias(ada_full_PWInteraction1_V2):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_bias, self).__init__(Mconver)
                bias = torch.zeros(self.in_planes).unsqueeze(-1).unsqueeze(-1)
                self.bias = nn.Parameter(bias)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                self.cache = self.cache + self.bn(F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights))
                self.start_ind = num_to_shift
                return self.relu(self.cache+self.bias)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        '''
        class ada_full_PWInteraction1_V2_detach(ada_full_PWInteraction1_V2):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_detach, self).__init__(Mconver)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                self.cache = self.cache + F.conv2d(cur_out[:, self.start_ind:num_to_shift].detach(), raw_weights)
                self.start_ind = num_to_shift
                return self.relu(self.cache)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction1_V2_SigMgate(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_SigMgate, self).__init__()
                self.start_ind = 0
                self.out_folds = Mconver.out_folds[:-1]
                #self.out_planes = Mconver.out_planes
                self.in_planes = Mconver.in_planes
                #self.planes = torch.tensor(self.out_folds).sum()
                self.planes = sum(self.out_folds)

                Weight_mat = torch.zeros(self.in_planes,self.planes).unsqueeze(-1).unsqueeze(-1)
                self.Weight_mat = nn.Parameter(Weight_mat)
                self.gate = nn.Sigmoid()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                self.cache = self.cache + F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights)
                self.g = self.gate(self.cache)
                self.start_ind = num_to_shift
                return self.g * sp_now
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction1_V2_ablation(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_ablation, self).__init__()
                self.out_folds = Mconver.out_folds[:-1]
                #self.out_planes = Mconver.out_planes
                self.in_planes = Mconver.in_planes
                #self.planes = torch.tensor(self.out_folds).sum()
                self.planes = sum(self.out_folds)
                Weight_mat = torch.eye(self.in_planes,self.in_planes).unsqueeze(-1).unsqueeze(-1)
                self.Weight_mat = nn.Parameter(Weight_mat)
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                self.cache = F.conv2d(sp_now, self.Weight_mat)
                return self.relu(self.cache)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction1_V2_woCache(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_woCache, self).__init__()
                self.start_ind = 0
                self.out_folds = Mconver.out_folds[:-1]
                #self.out_planes = Mconver.out_planes
                self.in_planes = Mconver.in_planes
                #self.planes = torch.tensor(self.out_folds).sum()
                self.planes = sum(self.out_folds)

                Weight_mat = torch.zeros(self.in_planes,self.planes).unsqueeze(-1).unsqueeze(-1)
                self.Weight_mat = nn.Parameter(Weight_mat)
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                self.cache = sp_now + F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights)
                self.start_ind = num_to_shift
                return self.relu(self.cache)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction1_V2_woCache_wBN(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_woCache_wBN, self).__init__()
                self.start_ind = 0
                self.out_folds = Mconver.out_folds[:-1]
                #self.out_planes = Mconver.out_planes
                self.in_planes = Mconver.in_planes
                #self.planes = torch.tensor(self.out_folds).sum()
                self.planes = sum(self.out_folds)
                self.start_ind2sp_ind = dict()
                start_inds = torch.tensor([0]+self.out_folds[:-1]).cumsum(dim=0)
                for i, ind in enumerate(start_inds):
                    self.start_ind2sp_ind[ind.item()] = i
                #
                self.affiners = []
                for _ in range(len(self.out_folds)):
                    self.affiners.append(BN_affine(self.in_planes, 1, 0))
                self.affiners = nn.Sequential(*self.affiners)

                Weight_mat = torch.zeros(self.in_planes,self.planes).unsqueeze(-1).unsqueeze(-1)
                self.Weight_mat = nn.Parameter(Weight_mat)
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                self.cache = self.affiners[self.start_ind2sp_ind[self.start_ind]](sp_now) + F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights)
                self.start_ind = num_to_shift
                return self.relu(self.cache)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction1_V3(nn.Module):
            # Add BNs for input 
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V3, self).__init__()
                self.start_ind = 0
                self.out_folds = Mconver.out_folds[:-1]
                #self.out_planes = Mconver.out_planes
                self.in_planes = Mconver.in_planes
                #self.planes = torch.tensor(self.out_folds).sum()
                self.planes = sum(self.out_folds)

                Weight_mat = torch.zeros(self.in_planes,self.planes).unsqueeze(-1).unsqueeze(-1)
                self.Weight_mat = nn.Parameter(Weight_mat)
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                self.cache = self.cache + F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights)

                self.start_ind = num_to_shift
                return self.relu(self.cache)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction1_V2_bn(ada_full_PWInteraction1_V2):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction1_V2_bn, self).__init__(Mconver)
                self.bn = nn.BatchNorm2d(Mconver.in_planes)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                self.cache = self.cache + self.bn(F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights))
                self.start_ind = num_to_shift
                return self.relu(self.cache)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction2(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2, self).__init__()
                self.out_folds = Mconver.out_folds[:-1]
                self.planes = torch.tensor(self.out_folds).sum()
                #Weight_mat = torch.zeros(self.planes, self.planes, 2, 1, 1).cuda()
                #Weight_mat[:, :, 0] = 0
                Weight_mat = torch.stack([torch.zeros(self.planes,self.planes), torch.eye(self.planes)], -1).unsqueeze(-1).unsqueeze(-1)
                self.Weight_mat = nn.Parameter(Weight_mat)
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                num_to_shift = cur_out.size(1).item()
                raw_weights = self.Weight_mat[:num_to_shift, :num_to_shift, 0]
                raw_weights2 = self.Weight_mat[:num_to_shift, :num_to_shift, 1]
                interacted_feat = F.conv2d(cur_out, raw_weights) + F.conv2d(sp_now[:, :num_to_shift], raw_weights2)
                interacted_feat = self.relu(interacted_feat)
                return torch.cat([interacted_feat, sp_now[:, num_to_shift:]], 1)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction2_V2(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2, self).__init__()
                self.start_ind = 0
                self.out_folds = Mconver.out_folds[:-1]
                self.planes = torch.tensor(self.out_folds).sum()
                self.in_planes = Mconver.in_planes
                #Weight_mat = torch.zeros(self.planes, self.planes, 2, 1, 1).cuda()
                #Weight_mat[:, :, 0] = 0
                Weight_mat0 = torch.zeros(self.in_planes, self.planes).unsqueeze(-1).unsqueeze(-1)
                self.Weight_mat0 = nn.Parameter(Weight_mat0)
                Weight_mat1 =  torch.eye(self.in_planes).unsqueeze(-1).unsqueeze(-1)
                self.Weight_mat1 = nn.Parameter(Weight_mat1)
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = F.conv2d(sp_now, self.Weight_mat1)
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                self.cache = self.cache + F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights)
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(self.cache)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction2_V2_CAttendByAdd(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_CAttendByAdd, self).__init__()
                self.start_ind = 0
                self.out_folds = Mconver.out_folds[:-1]
                self.planes = torch.tensor(self.out_folds).sum()
                self.in_planes = Mconver.in_planes
                #Weight_mat = torch.zeros(self.planes, self.planes, 2, 1, 1).cuda()
                #Weight_mat[:, :, 0] = 0
                self.atten_reduce_factor = 4
                self.Weight_mat_attend = nn.Parameter(torch.zeros(self.in_planes,self.planes).unsqueeze(-1).unsqueeze(-1))
                self.Weight_mat0 = nn.Parameter(torch.zeros(self.in_planes, self.planes).unsqueeze(-1).unsqueeze(-1))
                self.Weight_mat1 = nn.Parameter(torch.eye(self.in_planes).unsqueeze(-1).unsqueeze(-1))
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights_attend = self.Weight_mat_attend[:, self.start_ind:num_to_shift]
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                sp_past = cur_out[:, self.start_ind:num_to_shift]
                # attend
                self.cache_attend = self.cache_attend + F.conv2d(sp_past.mean((-1,-2),keepdim=True), raw_weights_attend)
                attened = self.relu(sp_now + self.cache_attend)
                # fuse
                fused = F.conv2d(attened, self.Weight_mat1)
                self.cache_add = self.cache_add + F.conv2d(sp_past, raw_weights)
                fused = fused + self.cache_add
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction2_V2_CAttendByAdd_outReLU(ada_full_PWInteraction2_V2_CAttendByAdd):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_CAttendByAdd_outReLU, self).__init__(Mconver)
                #nn.init.constant_(self.Weight_mat1, 0)
                #nn.init.kaiming_normal_(self.Weight_mat_attend)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                    self.cache_fused = F.conv2d(sp_now, self.Weight_mat1)
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights_attend = self.Weight_mat_attend[:, self.start_ind:num_to_shift]
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                sp_past = cur_out[:, self.start_ind:num_to_shift]
                # attend
                self.cache_attend = self.cache_attend + F.conv2d(sp_past.mean((-1,-2),keepdim=True), raw_weights_attend)
                attened = self.cache_attend
                # fuse
                fused = self.relu(self.cache_fused + F.conv2d(self.cache_attend, self.Weight_mat1))
                self.cache_add = self.cache_add + F.conv2d(sp_past, raw_weights)
                fused = fused + self.cache_add
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)

        class ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_Impl0(ada_full_PWInteraction2_V2_CAttendByAdd):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_Impl0, self).__init__(Mconver)
                self.num_segments = Mconver.args.num_segments
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                    self.cache_fused = F.conv2d(sp_now, self.Weight_mat1)

                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights_attend = self.Weight_mat_attend[:, self.start_ind:num_to_shift]
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                sp_past = cur_out[:, self.start_ind:num_to_shift]
                sp_past_st = sp_past.view((-1, self.num_segments) + (sp_past.size()[-3:]))
                sp_past_st = sp_past_st.mean(1,keepdim=True)
                sp_past_st_s = sp_past_st.view((-1,)+sp_past.size()[-3:])
                # attend
                self.cache_attend = self.cache_attend + F.conv2d(sp_past.mean((-1,-2),keepdim=True), raw_weights_attend)
                attened = self.cache_attend
                # fuse
                fused = self.relu(self.cache_fused + F.conv2d(self.cache_attend, self.Weight_mat1))
                self.cache_add = self.cache_add + F.conv2d(sp_past, raw_weights)
                fused = fused + self.cache_add
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU(ada_full_PWInteraction2_V2_CAttendByAdd):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU, self).__init__(Mconver)
                self.num_segments = Mconver.args.num_segments
                nn.init.constant_(self.Weight_mat1, 0)
                #nn.init.kaiming_normal_(self.Weight_mat_attend)
                nn.init.constant_(self.Weight_mat_attend, 0)
                #set_trace()
                layer = Mconver.layer_info[0][0]
                if layer=='S3d' or layer=='T':
                    self.Weight_mat1 = nn.Parameter(self.Weight_mat1.unsqueeze(2))
                    self.Weight_mat_attend = nn.Parameter(self.Weight_mat_attend.unsqueeze(2))
                    self.Weight_mat0 = nn.Parameter(self.Weight_mat0.unsqueeze(2))
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                    if sp_now.dim()==4:
                        self.conv_func = F.conv2d
                    elif sp_now.dim()==5:
                        self.conv_func = F.conv3d
                    else:
                        raise ValueError
                    '''
                    if sp_now.dim()!=self.Weight_mat1.dim():
                        self.Weight_mat1 = nn.Parameter(self.Weight_mat1.unsqueeze(2))
                        self.Weight_mat_attend = nn.Parameter(self.Weight_mat_attend.unsqueeze(2))
                        self.Weight_mat0 = nn.Parameter(self.Weight_mat0.unsqueeze(2))
                        set_trace()
                    '''
                    self.cache_fused = self.conv_func(sp_now, self.Weight_mat1)
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights_attend = self.Weight_mat_attend[:, self.start_ind:num_to_shift]
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                sp_past = cur_out[:, self.start_ind:num_to_shift]
                if sp_past.dim()==4:
                    
                    try:
                        sp_past_stView = sp_past.view((-1, self.num_segments) + (sp_past.size()[-3:]))
                    except:
                        set_trace()
                elif sp_past.dim()==5:
                    sp_past_stView = sp_past.transpose(1,2)
                # sp_past_stView of size (bs, t, c, h, w)
                sp_past_stView_TPooled = sp_past_stView.mean(1,keepdim=True)
                sp_past_stView_SPooled = sp_past_stView.mean((-1,-2),keepdim=True)
                if sp_past.dim()==4:
                    sp_past_Pooled = (sp_past_stView_TPooled+sp_past_stView_SPooled).view((-1,sp_past_stView_SPooled.size()[2])+sp_past_stView_TPooled.size()[-2:])
                    # sp_past_stView of size (bs*t, c, h, w)
                elif sp_past.dim()==5:
                    sp_past_Pooled = (sp_past_stView_TPooled+sp_past_stView_SPooled).transpose(1,2)
                    # sp_past_stView of size (bs, c, t, h, w)
                # attend
                self.cache_attend = self.cache_attend + self.conv_func(sp_past_Pooled, raw_weights_attend)
                attened = self.cache_attend
                # fuse
                #self.cache_fused=0
                #self.cache_attend=0
                fused = self.relu(sp_now + self.cache_fused + self.cache_attend)
                self.cache_add = self.cache_add + self.conv_func(sp_past, raw_weights)
                #self.cache_add=0
                fused = fused + self.cache_add
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)

        class ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_leaky11(ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_leaky11, self).__init__(Mconver)
                self.relu = nn.LeakyReLU(inplace=True)

        class ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_OnlyAttend(ada_full_PWInteraction2_V2_CAttendByAdd):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_OnlyAttend, self).__init__(Mconver)
                self.num_segments = Mconver.args.num_segments
                nn.init.constant_(self.Weight_mat1, 0)
                nn.init.constant_(self.Weight_mat_attend, 0)
                layer = Mconver.layer_info[0][0]
                if layer=='S3d' or layer=='T':
                    self.Weight_mat1 = nn.Parameter(self.Weight_mat1.unsqueeze(2))
                    self.Weight_mat_attend = nn.Parameter(self.Weight_mat_attend.unsqueeze(2))
                    self.Weight_mat0 = nn.Parameter(self.Weight_mat0.unsqueeze(2))
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                    if sp_now.dim()==4:
                        self.conv_func = F.conv2d
                    elif sp_now.dim()==5:
                        self.conv_func = F.conv3d
                    else:
                        raise ValueError
                    #self.cache_fused = self.conv_func(sp_now, self.Weight_mat1)
                    # abs
                    self.cache_fused =0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights_attend = self.Weight_mat_attend[:, self.start_ind:num_to_shift]
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                sp_past = cur_out[:, self.start_ind:num_to_shift]
                if sp_past.dim()==4:
                    sp_past_stView = sp_past.view((-1, self.num_segments) + (sp_past.size()[-3:]))
                elif sp_past.dim()==5:
                    sp_past_stView = sp_past.transpose(1,2)
                # sp_past_stView of size (bs, t, c, h, w)
                sp_past_stView_TPooled = sp_past_stView.mean(1,keepdim=True)
                sp_past_stView_SPooled = sp_past_stView.mean((-1,-2),keepdim=True)
                if sp_past.dim()==4:
                    sp_past_Pooled = (sp_past_stView_TPooled+sp_past_stView_SPooled).view((-1,sp_past_stView_SPooled.size()[2])+sp_past_stView_TPooled.size()[-2:])
                    # sp_past_stView of size (bs*t, c, h, w)
                elif sp_past.dim()==5:
                    sp_past_Pooled = (sp_past_stView_TPooled+sp_past_stView_SPooled).transpose(1,2)
                    # sp_past_stView of size (bs, c, t, h, w)
                # attend
                self.cache_attend = self.cache_attend + self.conv_func(sp_past_Pooled, raw_weights_attend)
                attened = self.cache_attend
                fused = self.relu(sp_now + self.cache_fused + self.cache_attend)
                #abs
                self.cache_add = self.cache_add #+ self.conv_func(sp_past, raw_weights)
                fused = fused + self.cache_add
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)

        class ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_OnlyFuse(ada_full_PWInteraction2_V2_CAttendByAdd):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_OnlyFuse, self).__init__(Mconver)
                self.num_segments = Mconver.args.num_segments
                nn.init.constant_(self.Weight_mat1, 0)
                nn.init.constant_(self.Weight_mat_attend, 0)
                layer = Mconver.layer_info[0][0]
                if layer=='S3d' or layer=='T':
                    self.Weight_mat1 = nn.Parameter(self.Weight_mat1.unsqueeze(2))
                    self.Weight_mat_attend = nn.Parameter(self.Weight_mat_attend.unsqueeze(2))
                    self.Weight_mat0 = nn.Parameter(self.Weight_mat0.unsqueeze(2))
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                    if sp_now.dim()==4:
                        self.conv_func = F.conv2d
                    elif sp_now.dim()==5:
                        self.conv_func = F.conv3d
                    else:
                        raise ValueError
                    #self.cache_fused = self.conv_func(sp_now, self.Weight_mat1)
                    # abs
                    self.cache_fused =0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights_attend = self.Weight_mat_attend[:, self.start_ind:num_to_shift]
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                sp_past = cur_out[:, self.start_ind:num_to_shift]
                
                fused = self.relu(sp_now + self.cache_fused + self.cache_attend)
                #abs
                self.cache_add = self.cache_add + self.conv_func(sp_past, raw_weights)
                fused = fused + self.cache_add
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)

        class ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_down2x(ada_full_PWInteraction2_V2_CAttendByAdd):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_down2x, self).__init__(Mconver)
                self.num_segments = Mconver.args.num_segments
                nn.init.constant_(self.Weight_mat1, 0)
                #nn.init.kaiming_normal_(self.Weight_mat_attend)
                nn.init.constant_(self.Weight_mat_attend, 0)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                #self.upsample = nn.Upsample(scale_factor=(2,2), mode='nearest')
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                    self.cache_fused = F.conv2d(sp_now, self.Weight_mat1)
                    self.upsample = nn.Upsample(size=(sp_now.size()[-1],sp_now.size()[-2]), mode='nearest')
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights_attend = self.Weight_mat_attend[:, self.start_ind:num_to_shift]
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                sp_past = cur_out[:, self.start_ind:num_to_shift]
                sp_past_stView = sp_past.view((-1, self.num_segments) + (sp_past.size()[-3:]))
                sp_past_stView_TPooled = sp_past_stView.mean(1,keepdim=True)
                sp_past_stView_SPooled = sp_past_stView.mean((-1,-2),keepdim=True)
                sp_past_Pooled = (sp_past_stView_TPooled+sp_past_stView_SPooled).view((-1,)+sp_past.size()[-3:])
                # attend
                self.cache_attend = self.cache_attend + F.conv2d(sp_past_Pooled, raw_weights_attend)
                attened = self.cache_attend
                # fuse
                fused = self.relu(sp_now + self.cache_fused + self.cache_attend)
                self.cache_add = self.cache_add + self.upsample(F.conv2d(self.maxpool(sp_past), raw_weights))
                try:
                    fused = fused + self.cache_add
                except:
                    set_trace()

                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_abs(ada_full_PWInteraction2_V2_CAttendByAdd):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_abs, self).__init__(Mconver)
                self.num_segments = Mconver.args.num_segments
                nn.init.constant_(self.Weight_mat1, 0)
                #nn.init.kaiming_normal_(self.Weight_mat_attend)
                nn.init.constant_(self.Weight_mat_attend, 0)
                #set_trace()
                layer = Mconver.layer_info[0][0]
                if layer=='S3d' or layer=='T':
                    self.Weight_mat1 = nn.Parameter(self.Weight_mat1.unsqueeze(2))
                    self.Weight_mat_attend = nn.Parameter(self.Weight_mat_attend.unsqueeze(2))
                    self.Weight_mat0 = nn.Parameter(self.Weight_mat0.unsqueeze(2))
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                    if sp_now.dim()==4:
                        self.conv_func = F.conv2d
                    elif sp_now.dim()==5:
                        self.conv_func = F.conv3d
                    else:
                        raise ValueError
                    '''
                    if sp_now.dim()!=self.Weight_mat1.dim():
                        self.Weight_mat1 = nn.Parameter(self.Weight_mat1.unsqueeze(2))
                        self.Weight_mat_attend = nn.Parameter(self.Weight_mat_attend.unsqueeze(2))
                        self.Weight_mat0 = nn.Parameter(self.Weight_mat0.unsqueeze(2))
                        set_trace()
                    '''
                    self.cache_fused = self.conv_func(sp_now, self.Weight_mat1)
                    #self.cache_fused = 0
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights_attend = self.Weight_mat_attend[:, self.start_ind:num_to_shift]
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                sp_past = cur_out[:, self.start_ind:num_to_shift]
                if sp_past.dim()==4:
                    sp_past_stView = sp_past.view((-1, self.num_segments) + (sp_past.size()[-3:]))
                elif sp_past.dim()==5:
                    sp_past_stView = sp_past.transpose(1,2)
                # sp_past_stView of size (bs, t, c, h, w)
                sp_past_stView_TPooled = sp_past_stView.mean(1,keepdim=True)
                sp_past_stView_SPooled = sp_past_stView.mean((-1,-2),keepdim=True)
                if sp_past.dim()==4:
                    sp_past_Pooled = (sp_past_stView_TPooled+sp_past_stView_SPooled).view((-1,sp_past_stView_SPooled.size()[2])+sp_past_stView_TPooled.size()[-2:])
                    # sp_past_stView of size (bs*t, c, h, w)
                elif sp_past.dim()==5:
                    sp_past_Pooled = (sp_past_stView_TPooled+sp_past_stView_SPooled).transpose(1,2)
                    # sp_past_stView of size (bs, c, t, h, w)
                # attend
                self.cache_attend = self.cache_attend + self.conv_func(sp_past_Pooled, raw_weights_attend)
                attened = self.cache_attend
                # fuse
                #self.cache_fused=0
                #self.cache_attend=0
                fused = self.relu(sp_now + self.cache_fused + self.cache_attend)
                # Original
                #self.cache_add = self.cache_add + self.conv_func(sp_past, raw_weights)
                # No raw_weights
                self.cache_add = self.cache_add
                #self.cache_add=0
                fused = fused + self.cache_add
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)

        class ada_full_PWInteraction2_V2_STPool_DualCAttendByAdd_outReLU(ada_full_PWInteraction2_V2_CAttendByAdd):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_STPool_DualCAttendByAdd_outReLU, self).__init__(Mconver)
                self.sigm = nn.Sigmoid()
                self.num_segments = Mconver.args.num_segments
                self.InAttendOut_Weight_mat = nn.Parameter(torch.zeros(self.planes, self.in_planes, 1, 1))
                nn.init.constant_(self.Weight_mat1, 0)
                #nn.init.kaiming_normal_(self.Weight_mat_attend)
                nn.init.constant_(self.Weight_mat_attend, 0)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                    self.cache_fused = F.conv2d(sp_now, self.Weight_mat1)
                    #set_trace()
                    #self.InAttendOut = self.sigm(F.conv2d(sp_now.mean((-1,-2),keepdim=True), self.InAttendOut_Weight_mat))*2
                    self.InAttendOut = F.conv2d(sp_now.mean((-1,-2),keepdim=True), self.InAttendOut_Weight_mat)
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights_attend = self.Weight_mat_attend[:, self.start_ind:num_to_shift]
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                # InAttendOut
                print(self.InAttendOut.max())
                sp_past = self.relu(cur_out[:, self.start_ind:num_to_shift] + self.InAttendOut[:, self.start_ind:num_to_shift])
                #set_trace()
                #sp_past = cur_out[:, self.start_ind:num_to_shift] * self.InAttendOut[:, self.start_ind:num_to_shift]
                #sp_past = cur_out[:, self.start_ind:num_to_shift]
                sp_past_stView = sp_past.view((-1, self.num_segments) + (sp_past.size()[-3:]))
                sp_past_stView_TPooled = sp_past_stView.mean(1,keepdim=True)
                sp_past_stView_SPooled = sp_past_stView.mean((-1,-2),keepdim=True)
                sp_past_Pooled = (sp_past_stView_TPooled+sp_past_stView_SPooled).view((-1,)+sp_past.size()[-3:])
                # attend
                self.cache_attend = self.cache_attend + F.conv2d(sp_past_Pooled, raw_weights_attend)
                attened = self.cache_attend
                # fuse
                fused = self.relu(sp_now + self.cache_fused + self.cache_attend)
                self.cache_add = self.cache_add + F.conv2d(sp_past, raw_weights)
                fused = fused + self.cache_add
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)


        class ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_abs2(ada_full_PWInteraction2_V2_CAttendByAdd):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_abs2, self).__init__(Mconver)
                self.num_segments = Mconver.args.num_segments
                nn.init.constant_(self.Weight_mat1, 0)
                #nn.init.kaiming_normal_(self.Weight_mat_attend)
                nn.init.constant_(self.Weight_mat_attend, 0)
                self.dropout = nn.Dropout2d(.5)
                self.sigm = nn.Sigmoid()
                #self.Weight_mat_fusion_gate = nn.Parameter(torch.zeros(self.planes, self.in_planes).unsqueeze(-1).unsqueeze(-1))
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                    # abs
                    #self.cache_fused = F.conv2d(sp_now, self.Weight_mat1)
                    self.cache_fused = 0
                    sp_past_SPooled = sp_now.mean((-1,-2),keepdim=True)
                    self.gates =  self.sigm(F.conv2d(sp_past_SPooled, self.Weight_mat_fusion_gate))
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights_attend = self.Weight_mat_attend[:, self.start_ind:num_to_shift]
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                gates = self.gates[:, self.start_ind:num_to_shift]
                sp_past = cur_out[:, self.start_ind:num_to_shift]
                sp_past_stView = sp_past.view((-1, self.num_segments) + (sp_past.size()[-3:]))
                sp_past_stView_TPooled = sp_past_stView.mean(1,keepdim=True)
                sp_past_stView_SPooled = sp_past_stView.mean((-1,-2),keepdim=True)
                #sp_past_Pooled = sp_past_stView_TPooled.mean((-1,-2),keepdim=True)
                # abs
                #sp_past_stView_SPooled[sp_past_stView_SPooled!=0] = 0
                #sp_past_Pooled = (sp_past_stView_TPooled+sp_past_stView_SPooled).view((-1,)+sp_past.size()[-3:])
                #sp_past_Pooled = sp_past_stView_SPooled.mean(1,keepdim=True).repeat_interleave(self.num_segments, dim=0).view((-1,sp_past.size()[-3])+(1,1))
                sp_past_Pooled = (sp_past_stView_SPooled).view((-1,)+(sp_past.size()[-3],)+(1,1))
                # attend
                self.cache_attend = self.cache_attend + F.conv2d(sp_past_Pooled, raw_weights_attend)
                #self.cache_attend = self.cache_attend + F.conv2d(sp_past_Pooled, raw_weights_attend).repeat_interleave(self.num_segments, dim=0)
                attened = self.cache_attend
                # fuse
                # abs
                #fused = self.relu(self.cache_fused + self.cache_attend)
                fused = self.relu(sp_now + self.cache_fused + self.cache_attend)
                # abs share param 
                #self.cache_add = self.cache_add + F.conv2d(sp_past, raw_weights_attend)
                # abs gate
                self.cache_add = self.cache_add + F.conv2d(sp_past*gates, raw_weights)
                # normal
                #self.cache_add = self.cache_add + F.conv2d(sp_past, raw_weights)
                fused = fused + self.cache_add
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)

        class ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_FLOPs(nn.Module):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_FLOPs, self).__init__()
                self.start_ind = 0
                self.out_folds = Mconver.out_folds[:-1]
                self.planes = torch.tensor(self.out_folds).sum()
                self.in_planes = Mconver.in_planes
                self.atten_reduce_factor = 4
                #self.Weight_mat_attend = nn.Parameter(torch.zeros(self.in_planes,self.planes).unsqueeze(-1).unsqueeze(-1))
                #self.Weight_mat0 = nn.Parameter(torch.zeros(self.in_planes, self.planes).unsqueeze(-1).unsqueeze(-1))
                #self.Weight_mat1 = nn.Parameter(torch.eye(self.in_planes).unsqueeze(-1).unsqueeze(-1))
                self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, groups=1, padding=0, bias=False)
                self.denses_attend = []
                self.denses_add = []
                self.c2i = {}
                c_cum = 0
                for i, c in enumerate(self.out_folds):
                    self.c2i[c_cum] = i
                    c_cum += c
                    self.denses_attend.append(nn.Conv2d(c, self.in_planes, kernel_size=1, groups=1, bias=False))
                    self.denses_add.append(nn.Conv2d(c, self.in_planes, kernel_size=1, groups=1, bias=False))
                self.denses_attend = nn.Sequential(*self.denses_attend)
                self.denses_add = nn.Sequential(*self.denses_add)
                self.relu = nn.ReLU(inplace=True)
                self.num_segments = Mconver.args.num_segments

            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                    self.cache_fused = self.conv1(sp_now)
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                sp_past = cur_out[:, self.start_ind:num_to_shift]
                sp_past_stView = sp_past.view((-1, self.num_segments) + (sp_past.size()[-3:]))
                sp_past_stView_TPooled = sp_past_stView.mean(1,keepdim=False)
                #sp_past_stView_SPooled = sp_past_stView.mean((-1,-2),keepdim=True)
                sp_past_SPooled = sp_past.mean((-1,-2),keepdim=True)
                #sp_past_Pooled = (sp_past_stView_TPooled+sp_past_stView_SPooled).view((-1,)+sp_past.size()[-3:])
                # attend
                self.cache_attend = self.cache_attend + self.denses_add[self.c2i[self.start_ind]](sp_past_SPooled) + self.denses_add[self.c2i[self.start_ind]](sp_past_stView_TPooled).repeat_interleave(self.num_segments, dim=0)
                # fuse
                fused = self.relu(self.cache_fused + self.cache_attend)
                self.cache_add = self.cache_add + self.denses_add[self.c2i[self.start_ind]](sp_past)
                fused = fused + self.cache_add
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)

        class ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_reparam(ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_reparam, self).__init__(Mconver)
                self.relu = nn.Identity()

        class ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_Impl2(ada_full_PWInteraction2_V2_CAttendByAdd):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_Impl2, self).__init__(Mconver)
                self.num_segments = Mconver.args.num_segments
                nn.init.constant_(self.Weight_mat1, 0)
                #nn.init.kaiming_normal_(self.Weight_mat_attend)
                nn.init.constant_(self.Weight_mat_attend, 0)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                    self.cache_fused = F.conv2d(sp_now, self.Weight_mat1)
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights_attend = self.Weight_mat_attend[:, self.start_ind:num_to_shift]
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                sp_past = cur_out[:, self.start_ind:num_to_shift]
                sp_past_stView = sp_past.view((-1, self.num_segments) + (sp_past.size()[-3:]))
                sp_past_stView_TPooled = sp_past_stView.mean(1,keepdim=True)
                sp_past_stView_SPooled = sp_past_stView.mean((-1,-2),keepdim=True)
                sp_past_Pooled = (sp_past_stView_TPooled+sp_past_stView_SPooled).view((-1,)+sp_past.size()[-3:])
                # attend
                self.cache_attend = self.cache_attend + F.conv2d(torch.cat([sp_past_Pooled,sp_past],1), torch.cat([raw_weights_attend,raw_weights],1))
                # fuse
                fused = sp_now + self.cache_attend
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)

        class STPool_CAttendByAddReLU(nn.Module):
            def __init__(self, Mconver):
                super(STPool_CAttendByAddReLU, self).__init__()
                self.num_segments = Mconver.args.num_segments
                self.start_ind = 0
                self.out_folds = Mconver.out_folds[:-1]
                self.planes = torch.tensor(self.out_folds).sum()
                self.in_planes = Mconver.in_planes
                self.atten_reduce_factor = 4
                self.Weight_mat0S = nn.Parameter(torch.zeros(self.in_planes, self.planes).unsqueeze(-1).unsqueeze(-1))
                self.Weight_mat0T = nn.Parameter(torch.zeros(self.in_planes, self.planes).unsqueeze(-1).unsqueeze(-1))
                self.Weight_mat1S = nn.Parameter(torch.zeros(self.in_planes, self.in_planes).unsqueeze(-1).unsqueeze(-1))
                self.Weight_mat1T = nn.Parameter(torch.zeros(self.in_planes, self.in_planes).unsqueeze(-1).unsqueeze(-1))
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights0S = self.Weight_mat0S[:, self.start_ind:num_to_shift]
                raw_weights0T = self.Weight_mat0T[:, self.start_ind:num_to_shift]
                if self.start_ind==0:
                    sp_past = cur_out
                    concated = torch.cat([sp_now, sp_past], 1)
                    concated_tpooled = concated.view((-1, self.num_segments) + (concated.size()[-3:])).mean(1)
                    #self.cache_attend = 0
                    self.cache_attend = F.conv2d(concated.mean((-1,-2),keepdim=True), torch.cat([self.Weight_mat1S,raw_weights0S], 1)) #+ \
                    #F.conv2d(concated_tpooled, torch.cat([self.Weight_mat1T,raw_weights0T], 1)).repeat_interleave(self.num_segments, dim=0)
                    return self.relu(sp_now + self.cache_attend)

                sp_past = cur_out[:, self.start_ind:num_to_shift]
                sp_past_tpooled = sp_past.view((-1, self.num_segments) + (sp_past.size()[-3:])).mean(1)
                # attend
                self.cache_attend = self.cache_attend + F.conv2d(sp_past.mean((-1,-2),keepdim=True), raw_weights0S) # + F.conv2d(sp_past_tpooled, raw_weights0T).repeat_interleave(self.num_segments, dim=0)
                return self.relu(sp_now + self.cache_attend)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)

        class ada_full_PWInteraction2_V2_CAttendByAdd_outReLU_woCache(ada_full_PWInteraction2_V2_CAttendByAdd):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_CAttendByAdd_outReLU_woCache, self).__init__(Mconver)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_fused = F.conv2d(sp_now, self.Weight_mat1)
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights_attend = self.Weight_mat_attend[:, self.start_ind:num_to_shift]
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                sp_past = cur_out[:, self.start_ind:num_to_shift]
                # attend
                self.cache_attend =  F.conv2d(sp_past.mean((-1,-2),keepdim=True), raw_weights_attend)
                attened = self.cache_attend
                # fuse, when approximately reparameterized, becomes self.relu(self.cache_fused + a constant tensor)
                fused = self.relu(self.cache_fused + F.conv2d(self.cache_attend, self.Weight_mat1))
                # when approximately reparameterized, becomes a constant tensor
                self.cache_add =  F.conv2d(sp_past, raw_weights)
                fused = fused + self.cache_add
                # So, in summary, a reparameterized ARC in inference-time only cost 'two feature map summations and twice relus w/o any resolution reductions'
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)

        class ada_full_PWInteraction2_V2_CAttendByAdd_outReLU_bn(ada_full_PWInteraction2_V2_CAttendByAdd):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_CAttendByAdd_outReLU_bn, self).__init__(Mconver)
                #set bns
                self.bns = []
                self.c2i = {}
                c_cum = 0
                for i, c in enumerate(self.out_folds):
                    self.c2i[c_cum] = i
                    c_cum += c
                    self.bns.append(nn.BatchNorm2d(self.in_planes))
                    nn.init.constant_(self.bns[-1].weight, 0)
                self.bns = nn.Sequential(*self.bns)
                #nn.init.kaiming_normal_(self.Weight_mat1)
                nn.init.kaiming_normal_(self.Weight_mat_attend)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                    self.cache_fused = F.conv2d(sp_now, self.Weight_mat1)
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights_attend = self.Weight_mat_attend[:, self.start_ind:num_to_shift]
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                sp_past = cur_out[:, self.start_ind:num_to_shift]
                # attend
                self.cache_attend = self.cache_attend + F.conv2d(sp_past.mean((-1,-2),keepdim=True), raw_weights_attend)
                attened = self.cache_attend
                # fuse
                fused = self.relu(self.cache_fused + self.bns[self.c2i[self.start_ind]](F.conv2d(self.cache_attend, self.Weight_mat1)))
                self.cache_add = self.cache_add + F.conv2d(sp_past, raw_weights)
                fused = fused + self.cache_add
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)

        class ada_full_PWInteraction2_V2_CAttendByAdd_outReLU_MLP(ada_full_PWInteraction2_V2_CAttendByAdd_outReLU_bn):
            def __init__(self, Mconver):
                super(ada_full_PWInteraction2_V2_CAttendByAdd_outReLU_MLP, self).__init__(Mconver)
                self.intermediate_dim = self.in_planes // Mconver.args.reduce_attend
                self.Weight_mat_attend = nn.Parameter(torch.zeros(self.intermediate_dim, self.planes).unsqueeze(-1).unsqueeze(-1))
                self.Weight_mat_attend_2 = nn.Parameter(torch.zeros(self.in_planes, self.intermediate_dim).unsqueeze(-1).unsqueeze(-1))
                nn.init.kaiming_normal_(self.Weight_mat_attend)
                nn.init.kaiming_normal_(self.Weight_mat_attend_2)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache_attend = 0
                    self.cache_add = 0
                    self.cache_fused = F.conv2d(sp_now, self.Weight_mat1)
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights_attend = self.Weight_mat_attend[:, self.start_ind:num_to_shift]
                raw_weights = self.Weight_mat0[:, self.start_ind:num_to_shift]
                sp_past = cur_out[:, self.start_ind:num_to_shift]
                # attend
                self.cache_attend = self.cache_attend + \
                	F.conv2d(
                	self.relu(
                	F.conv2d(
                	sp_past.mean((-1,-2),keepdim=True), raw_weights_attend)), self.Weight_mat_attend_2)
                attened = self.cache_attend
                # fuse
                fused = self.relu(self.cache_fused + self.bns[self.c2i[self.start_ind]](F.conv2d(self.cache_attend, self.Weight_mat1)))
                self.cache_add = self.cache_add + F.conv2d(sp_past, raw_weights)
                fused = fused + self.cache_add
                if self.start_ind==self.planes:
                    self.start_ind = 0
                else:
                    self.start_ind = num_to_shift
                return self.relu(fused)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)

        class ada_full_PWInteraction3(nn.Module):
            # designed for the Basicblock when crossing stages
            def __init__(self, Mconver, c_interact_flag):
                super(ada_full_PWInteraction3, self).__init__()
                self.c_interact_flag = c_interact_flag
                self.out_folds = Mconver.out_folds[:-1]
                self.planes = torch.tensor(self.out_folds).sum()
                self.in_planes = Mconver.in_planes
                self.whole_planes = torch.tensor(Mconver.out_folds).sum()
                Weight_mat_out = torch.zeros(self.in_planes, self.planes).unsqueeze(-1).unsqueeze(-1)
                if self.c_interact_flag:
                    Weight_mat_in = torch.eye(self.whole_planes).unsqueeze(-1).unsqueeze(-1)
                    self.Weight_mat_in = nn.Parameter(Weight_mat_in)
                    self.cache = None
                self.Weight_mat_out = nn.Parameter(Weight_mat_out)
                self.gate = nn.Identity()
                self.relu = nn.ReLU(inplace=True)
                
            def _channel_shift(self, cur_out, sp_now):
                num_to_shift = cur_out.size(1).item()
                raw_weights = self.Weight_mat[:num_to_shift, :num_to_shift, 0]
                raw_weights2 = self.Weight_mat[:num_to_shift, :num_to_shift, 1]
                if self.c_interact_flag:
                    interacted_feat = F.conv2d(cur_out, self.Weight_mat_out[self.in_planes, :num_to_shift]) + F.conv2d(sp_now, self.Weight_mat_in)
                else:
                    interacted_feat = F.conv2d(cur_out, self.Weight_mat_out[self.in_planes, :num_to_shift]) + sp_now
                interacted_feat = self.relu(interacted_feat)
                return interacted_feat
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class full_shifter(nn.Module):
            def __init__(self):
                super(full_shifter, self).__init__()
                self.start_ind = 0
            def _shift_slice(self, x, num_to_shift):
                C = x.size(1).item()
                #print(self.start_ind)
                if self.start_ind+num_to_shift > C-1:
                    #..(*discard*)..exceed..... start_ind ..(*discard*)..-1
                    exceed = (self.start_ind+num_to_shift) - (C-1)
                    ret = x[:, exceed:self.start_ind]
                else:
                    # 0 ......... start_ind ..(*discard*)..start_ind+num_to_shift....-1
                    ret = torch.cat([x[:,:self.start_ind], x[:,self.start_ind+num_to_shift:]], 1)
                self.start_ind += num_to_shift
                if self.start_ind > C-1:
                    self.start_ind = exceed
                return ret
            def _channel_shift(self, cur_out, sp_now):
                num_to_shift = cur_out.size(1).item()
                interacted_feat = cur_out
                return torch.cat([interacted_feat, self._shift_slice(sp_now, num_to_shift)], 1)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class res2net_style(nn.Module):
            def __init__(self):
                super(res2net_style, self).__init__()
            def forward(self, out, sp_past, sp_now):
                if out.size(1) > sp_now.size(1):
                    #raise ValueError('num of out channels should not be greater than the one of sp_now.')
                    return out[:,:sp_now.size(1)] + sp_now
                #set_trace()
                return torch.cat([out + sp_now[:,:out.size(1)], sp_now[:,out.size(1):]], 1)
        class res2net_style_SepJitBN(nn.Module):
            def __init__(self, Mconver):
                super(res2net_style_SepJitBN, self).__init__()
                #assert Mconver.with_bn==False, 'res2net_style_SepBN can only be applied when with_bn is False'
                self.out_folds = Mconver.out_folds
                self.in_planes = Mconver.in_planes
                self.num_weights = []
                for i,fold in enumerate(Mconver.out_folds):
                    if fold>self.in_planes:
                        self.out_folds[i] = self.in_planes
                    self.num_weights.append(self.out_folds[i])
                # for inspect convenience, do not use BN_affine
                self.num_weights = torch.cumsum(torch.tensor(self.num_weights), 0)
                self.num = torch.cumsum(self.num_weights, 0)[-2]
                self.w = nn.Parameter(torch.ones(self.num, 1, 1)*1e-8)
                self.b = nn.Parameter(torch.ones(self.num, 1, 1)*1e-8)
                self.expanded = False
                self.start_ind = 0
                self.gate = nn.Identity()
            def forward(self, out, sp_past, sp_now):
                num_to_shift = torch.tensor(out.size(1)).detach().item()
                # compatible when crossing stages
                num_to_shift = num_to_shift if num_to_shift<self.in_planes else self.in_planes
                sliced_w, sliced_b = self.w[self.start_ind:self.start_ind+num_to_shift], self.b[self.start_ind:self.start_ind+num_to_shift]
                self.start_ind = num_to_shift
                if out.size(1) > sp_now.size(1):
                    return affine(out[:,:sp_now.size(1)], sliced_w, sliced_b) + sp_now
                #set_trace()
                #return sp_now
                #return torch.cat([out + sp_now[:,:out.size(1)], sp_now[:,out.size(1):]],  1)
                return torch.cat([affine(out,sliced_w,sliced_b) + sp_now[:,:out.size(1)], sp_now[:,out.size(1):]],  1)
        class res2net_style_SepBN(nn.Module):
            def __init__(self, Mconver):
                super(res2net_style_SepBN, self).__init__()
                #assert Mconver.with_bn==False, 'res2net_style_SepBN can only be applied when with_bn is False'
                self.out_folds = Mconver.out_folds
                self.in_planes = Mconver.in_planes
                self.num_weights = []
                for i,fold in enumerate(Mconver.out_folds):
                    if fold>self.in_planes:
                        self.out_folds[i] = self.in_planes
                    self.num_weights.append(self.out_folds[i])
                self.num_weights = torch.cumsum(torch.tensor(self.num_weights), 0)
                self.bns = []
                for i in self.num_weights[:-1]:
                    planes = i.item() if i.item()<=self.in_planes else self.in_planes
                    self.bns.append(nn.BatchNorm2d(planes))
                    nn.init.constant_(self.bns[-1].weight, 1e-8)
                    #nn.init.constant_(self.bns[-1].weight, 1.)
                    nn.init.constant_(self.bns[-1].bias, 1e-8)
                self.bns = nn.Sequential(*self.bns)
                self.start_ind = 0
                self.gate = nn.ReLU(inplace=True)
                self.gate = nn.Identity()
            def forward(self, out, sp_past, sp_now):
                if out.size(1) > sp_now.size(1):
                    ret = self.gate(self.bns[self.start_ind](out[:,:sp_now.size(1)])) + sp_now
                else:
                    #set_trace()
                    ret = torch.cat([self.gate(self.bns[self.start_ind](out))+sp_now[:,:out.size(1)], sp_now[:,out.size(1):]],  1)
                self.start_ind += 1
                return ret
        class res2net_style_SepBN_InBN(nn.Module):
            def __init__(self, Mconver):
                super(res2net_style_SepBN_InBN, self).__init__()
                #assert Mconver.with_bn==False, 'res2net_style_SepBN can only be applied when with_bn is False'
                self.out_folds = Mconver.out_folds
                self.in_planes = Mconver.in_planes
                self.num_weights = []
                for i,fold in enumerate(Mconver.out_folds):
                    if fold>self.in_planes:
                        self.out_folds[i] = self.in_planes
                    self.num_weights.append(self.out_folds[i])
                self.num_weights = torch.cumsum(torch.tensor(self.num_weights), 0)
                self.bns = []
                self.In_bns = []
                for i in self.num_weights[:-1]:
                    planes = i.item() if i.item()<=self.in_planes else self.in_planes
                    self.bns.append(nn.BatchNorm2d(planes))
                    nn.init.constant_(self.bns[-1].weight, 1e-8)
                    #nn.init.constant_(self.bns[-1].weight, 1.)
                    nn.init.constant_(self.bns[-1].bias, 1e-8)
                    #***********In_BN
                    self.In_bns.append(nn.BatchNorm2d(self.in_planes))
                    nn.init.constant_(self.In_bns[-1].weight, 1)
                    nn.init.constant_(self.In_bns[-1].bias, 1e-8)
                self.bns = nn.Sequential(*self.bns)
                self.In_bns = nn.Sequential(*self.In_bns)
                self.start_ind = 0
                self.gate = nn.ReLU(inplace=True)
                self.gate = nn.Identity()
            def forward(self, out, sp_past, sp_now):
                sp_now = self.In_bns[self.start_ind](sp_now)
                if out.size(1) > sp_now.size(1):
                    ret = self.gate(self.bns[self.start_ind](out[:,:sp_now.size(1)])) + sp_now
                else:
                    #set_trace()
                    ret = torch.cat([self.gate(self.bns[self.start_ind](out))+sp_now[:,:out.size(1)], sp_now[:,out.size(1):]],  1)
                self.start_ind += 1
                return ret
        class res2net_style_SepBN_dense1(nn.Module):
            # It implements dense -> BN with weights init with 1 <=> PWI1+BN
            def __init__(self, Mconver):
                super(res2net_style_SepBN_dense1, self).__init__()
                self.start_ind = 0
                self.out_folds = Mconver.out_folds[:-1]
                self.in_planes = Mconver.in_planes
                self.planes = sum(self.out_folds)
                Weight_mat = torch.zeros(self.in_planes,self.planes).unsqueeze(-1).unsqueeze(-1)
                #Weight_mat = torch.randn(self.in_planes,self.planes).unsqueeze(-1).unsqueeze(-1)
                self.Weight_mat = nn.Parameter(Weight_mat)
                #
                self.start_ind2sp_ind = dict()
                start_inds = torch.tensor([0]+self.out_folds).cumsum(dim=0)
                for i, ind in enumerate(start_inds):
                    self.start_ind2sp_ind[ind.item()] = i
                self.bns = []
                for i in range(len(Mconver.out_folds)):
                    self.bns.append(nn.BatchNorm2d(self.in_planes))
                    #self.bns.append(nn.BatchNorm2d(Mconver.out_folds[i]))
                    #nn.init.constant_(self.bns[-1].weight, 1e-8)
                    nn.init.constant_(self.bns[-1].weight, 1)
                    nn.init.constant_(self.bns[-1].bias, 1e-8)
                self.bns = nn.Sequential(*self.bns)
                #
                self.gate = nn.Identity()
                #self.gate = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                #set_trace()
                # reduced to PWI1
                self.cache = self.cache + F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights)
                #self.cache = self.cache + self.bns[self.start_ind2sp_ind[self.start_ind]](F.conv2d(cur_out[:, self.start_ind:num_to_shift], raw_weights))
                #self.cache = self.cache + F.conv2d(self.bns[self.start_ind2sp_ind[self.start_ind]](cur_out[:, self.start_ind:num_to_shift]), raw_weights)
                self.start_ind = num_to_shift
                return self.gate(self.cache)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class res2net_style_SepJitBN_dense1(nn.Module):
            def __init__(self, Mconver):
                super(res2net_style_SepJitBN_dense1, self).__init__()
                self.start_ind = 0
                self.out_folds = Mconver.out_folds[:-1]
                self.in_planes = Mconver.in_planes
                self.planes = sum(self.out_folds)
                Weight_mat = torch.zeros(self.in_planes,self.planes).unsqueeze(-1).unsqueeze(-1)
                self.Weight_mat = nn.Parameter(Weight_mat)
                # set affiners
                self.start_ind2sp_ind = dict()
                start_inds = torch.tensor([0]+self.out_folds).cumsum(dim=0)
                for i, ind in enumerate(start_inds):
                    self.start_ind2sp_ind[ind.item()] = i
                self.bns = []
                for i in range(len(Mconver.out_folds)):
                    self.bns.append(jit.script(BN_affine(Mconver.out_folds[i], 1e-8, 1e-8)))
                    #self.bns.append(BN_affine(Mconver.out_folds[i], 1e-8, 1e-8))
                self.bns = nn.Sequential(*self.bns)
                #
                self.gate = nn.Identity()
                #self.gate = nn.ReLU(inplace=True)
            def _channel_shift(self, cur_out, sp_now):
                if self.start_ind==0:
                    self.cache = sp_now
                num_to_shift = torch.tensor(cur_out.size(1)).detach().item()
                raw_weights = self.Weight_mat[:, self.start_ind:num_to_shift]
                self.cache = self.cache + F.conv2d(self.bns[self.start_ind2sp_ind[self.start_ind]](cur_out[:, self.start_ind:num_to_shift]), raw_weights)
                self.start_ind = num_to_shift
                return self.gate(self.cache)
            def forward(self, out, sp_past, sp_now):
                return self._channel_shift(out, sp_now)
        class donothing(nn.Module):
            def __init__(self):
                super(donothing, self).__init__()
            def forward(self, out, sp_past, sp_now):
                return sp_now
        # if no affines in BNs, set external affines
        if self.no_affine:
            self.affiner = jit.script(BN_affine(self.out_planes, 1, 1e-8))
            #self.affiner = BN_affine(self.out_planes, 1, 1e-8)
        else:
            self.affiner = nn.Identity()
        self.layer = list()
        for i, info in enumerate(self.layer_info):
            conv_type, propotion = ''.join(info[:-1]), info[:-1]
            #set_trace()
            # hardcode kernel_size to 3
            kernel_size = 3
            if conv_type=='S':
                self.layer.append(Conv2d_bn_relu(in_planes, self.out_folds[i], kernel_size=(kernel_size, kernel_size), stride=stride,
                         padding=((kernel_size-1)//2, (kernel_size-1)//2), groups=groups, bias=bias, with_bn=self.with_bn, no_affine=self.no_affine, dilation=dilation, isReLU=isReLU))
            elif conv_type=='S3d':
                self.layer.append(Conv3d_bn_relu(in_planes, self.out_folds[i], kernel_size=(1, kernel_size, kernel_size), stride=(1,stride,stride),
                         padding=(0, (kernel_size-1)//2, (kernel_size-1)//2), groups=groups, bias=bias, with_bn=self.with_bn, no_affine=self.no_affine, dilation=dilation, isReLU=isReLU))
            elif conv_type=='H':
                self.layer.append(Conv2d_bn_relu(in_planes, self.out_folds[i], kernel_size=(kernel_size, 1), stride=stride,
                         padding=((kernel_size-1)//2, 0), groups=groups, bias=bias, with_bn=self.with_bn, no_affine=self.no_affine, dilation=dilation, isReLU=isReLU))
            elif conv_type=='W':
                self.layer.append(Conv2d_bn_relu(in_planes, self.out_folds[i], kernel_size=(1, kernel_size), stride=stride,
                         padding=(0, (kernel_size-1)//2), groups=groups, bias=bias, with_bn=self.with_bn, no_affine=self.no_affine, dilation=dilation, isReLU=isReLU))
            elif conv_type=='ST':
                self.layer.append(Conv3d_bn_relu(in_planes, self.out_folds[i], kernel_size=(kernel_size, kernel_size, kernel_size), stride=(1, stride, stride),
                         padding=((kernel_size-1)//2, (kernel_size-1)//2, (kernel_size-1)//2), groups=groups, bias=bias, with_bn=self.with_bn, no_affine=self.no_affine, dilation=dilation, isReLU=isReLU))
            elif conv_type=='T':
                # For temporal conv. the stride is hardcoded to 1.
                # 1D implementation is not comppatient with the spatial stide!=1 cases.
                #self.layer.append(Conv1d_bn_relu(in_planes, self.out_folds[i], kernel_size=kernel_size, stride=(1),
                #         padding=(kernel_size-1)//2, groups=groups, bias=bias, dilation=dilation))
                self.layer.append(Conv3d_bn_relu(in_planes, self.out_folds[i], kernel_size=(kernel_size, 1, 1), stride=(1, stride, stride),
                         padding=((kernel_size-1)//2, 0, 0), groups=groups, bias=bias, with_bn=self.with_bn, no_affine=self.no_affine, dilation=dilation, isReLU=isReLU))
            elif conv_type=='csT':
                self.layer.append(Conv3d_bn_relu(in_planes, self.out_folds[i], kernel_size=(kernel_size, 1, 1), stride=(1, stride, stride),
                         padding=((kernel_size-1)//2, 0, 0), groups=self.out_folds[i], bias=bias, with_bn=self.with_bn, no_affine=self.no_affine, dilation=dilation, isReLU=isReLU))
            else:
                raise NotImplementedError('conv_type: '+conv_type)
            # if 'SepBN' in self.interacter, register forward hook on conv moudle to hook its output
            if 'SepBN' in self.interacter:
                self.layer[-1].block[0].register_forward_hook(self.hook_fun)
            if 'reparam' in self.interacter:
                self.layer[-1].block[1].register_forward_hook(self.hook_fun)
        # for mixed precision training compatibility
        self.layer = nn.Sequential(*self.layer)
        #*********************channel_interacter**********************************************************
        #set_trace()
        if not (self.insert_propotion=='part' and self.stride!=1) or (self.insert_propotion=='all' and self.stride!=1):
            if args.interacter=='res2netStyle':
                #set_trace()
                self.channel_interacter = res2net_style()
            elif args.interacter=='res2net_style_SepBN':
                self.channel_interacter = res2net_style_SepBN(self)
            elif args.interacter=='res2net_style_SepBN_InBN':
                self.channel_interacter = res2net_style_SepBN_InBN(self)
            elif args.interacter=='res2net_style_SepJitBN':
                self.channel_interacter = res2net_style_SepJitBN(self)
            elif args.interacter=='res2net_style_SepBN_dense1':
                self.channel_interacter = res2net_style_SepBN_dense1(self)
            elif args.interacter=='res2net_style_SepJitBN_dense1':
                self.channel_interacter = res2net_style_SepJitBN_dense1(self)
            elif args.interacter=='PWI1':
                self.channel_interacter = ada_full_PWInteraction1_V2(self)
            elif args.interacter=='PWI1_pool_CAttend_add_KeepSpatial':
                self.channel_interacter = ada_full_PWInteraction1_V2_pool_CAttend_add_KeepSpatial(self)
            elif args.interacter=='PWI1_pool_CAttendByAdd_KeepSpatial':
                self.channel_interacter = ada_full_PWInteraction1_V2_pool_CAttendByAdd_KeepSpatial(self)
            elif args.interacter=='PWI1_pool_CAttendByAdd_KeepSpatialDWConv':
                self.channel_interacter = ada_full_PWInteraction1_V2_pool_CAttendByAdd_KeepSpatialDWConv(self)
            elif args.interacter=='PWI1_STpool_CAttendByAdd_KeepSpatialDWConv':
                self.channel_interacter = ada_full_PWInteraction1_V2_STpool_CAttendByAdd_KeepSpatialDWConv(self)
            elif args.interacter=='PWI1_pool_CAttendByAdd_KeepSpatialDWConv_FLOPs':
                self.channel_interacter = ada_full_PWInteraction1_V2_pool_CAttendByAdd_KeepSpatialDWConv_FLOPs(self)
            elif args.interacter=='PWI1_pool_CAttendByAdd_KeepSpatial_ForceDiverse':
                self.channel_interacter = ada_full_PWInteraction1_V2_pool_CAttendByAdd_KeepSpatial_ForceDiverse(self)
            elif args.interacter=='PWI1_SepPoolPW':
                self.channel_interacter = ada_full_PWInteraction1_V2_SepPoolPW(self)
            elif args.interacter=='PWI1_RNNPoolPW':
                self.channel_interacter = ada_full_PWInteraction1_V2_RNNPoolPW(self)
            elif args.interacter=='PWI1_dwConv':
                self.channel_interacter = ada_full_PWInteraction1_V2_dwConv(self)
            elif args.interacter=='PWI1_dwConv2':
                self.channel_interacter = ada_full_PWInteraction1_V2_dwConv2(self)
            elif args.interacter=='PWI1_dwConv2BN':
                self.channel_interacter = ada_full_PWInteraction1_V2_dwConv2BN(self)
            elif args.interacter=='PWI1_pool':
                self.channel_interacter = ada_full_PWInteraction1_V2_pool(self)
            elif args.interacter=='PWI1_pool_CAttend':
                self.channel_interacter = da_full_PWInteraction1_V2a_pool_CAttend(self)   
            elif args.interacter=='PWI1_pool_CAttend_add':
                self.channel_interacter = ada_full_PWInteraction1_V2_pool_CAttend_add(self)  
            elif args.interacter=='PWI1_pool_CAttend_add_woCache':
                self.channel_interacter = ada_full_PWInteraction1_V2_pool_CAttend_add_woCache(self)  
            elif args.interacter=='PWI1_CAttend_SharedDWConvadd_woCache':
                self.channel_interacter = ada_full_PWInteraction1_V2_pool_CAttend_SharedDWConvadd(self)  
            elif args.interacter=='PWI1_CAttend_SharedDWConvadd_woCache2':
                self.channel_interacter = ada_full_PWInteraction1_V2_pool_CAttend_SharedDWConvadd2(self)
            elif args.interacter=='PWI1_CAttend_SharedDWConvadd_woCache2_bn':
                self.channel_interacter = ada_full_PWInteraction1_V2_pool_CAttend_SharedDWConvadd2_bn(self)
            elif args.interacter=='PWI1_woReLU':
                self.channel_interacter = ada_full_PWInteraction1_V2_woReLU(self)
            elif args.interacter=='PWI1_bias':
                self.channel_interacter = ada_full_PWInteraction1_V2_bias(self)
            elif args.interacter=='PWI1_MLP2':
                self.channel_interacter = ada_full_PWInteraction1_V2_MLP2(self)
            elif args.interacter=='PWI1_detach':
                self.channel_interacter = ada_full_PWInteraction1_V2_detach(self)
            elif args.interacter=='PWI1_FLOPs':
                self.channel_interacter = ada_full_PWInteraction1_V2_FLOPs(self)
            elif args.interacter=='PWI1_bn':
                self.channel_interacter = ada_full_PWInteraction1_V2_bn(self)
            elif args.interacter=='PWI1_SigMgate':
                self.channel_interacter = ada_full_PWInteraction1_V2_SigMgate(self)
            elif args.interacter=='PWI1_abl':
                self.channel_interacter = ada_full_PWInteraction1_V2_ablation(self)
            elif args.interacter=='PWI1_woCache':
                self.channel_interacter = ada_full_PWInteraction1_V2_woCache(self)
            elif args.interacter=='PWI1_woCache_wBN':
                self.channel_interacter = ada_full_PWInteraction1_V2_woCache_wBN(self)
            elif args.interacter=='PWI2':
                self.channel_interacter = ada_full_PWInteraction2_V2(self)
            elif args.interacter=='PWI2_CAttendByAdd':
                self.channel_interacter = ada_full_PWInteraction2_V2_CAttendByAdd(self)
            elif args.interacter=='PWI2_CAttendByAdd_outReLU':
                self.channel_interacter = ada_full_PWInteraction2_V2_CAttendByAdd_outReLU(self)
            elif args.interacter=='PWI2_STPool_CAttendByAdd_outReLU':
                self.channel_interacter = ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU(self)
            elif args.interacter=='PWI2_STPool_CAttendByAdd_outReLU_leaky11':
                self.channel_interacter = ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_leaky11(self)
            elif args.interacter=='PWI2_STPool_CAttendByAdd_outReLU_OnlyAttend':
                self.channel_interacter = ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_OnlyAttend(self)
            elif args.interacter=='PWI2_STPool_CAttendByAdd_outReLU_OnlyFuse':
                self.channel_interacter = ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_OnlyFuse(self)
            elif args.interacter=='PWI2_STPool_DualCAttendByAdd_outReLU':
                self.channel_interacter = ada_full_PWInteraction2_V2_STPool_DualCAttendByAdd_outReLU(self)
            elif args.interacter=='PWI2_STPool_CAttendByAdd_outReLU_abs':
                self.channel_interacter = ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_abs(self)
            elif args.interacter=='PWI2_STPool_CAttendByAdd_outReLU_down2x':
                self.channel_interacter = ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_down2x(self)
            elif args.interacter=='PWI2_STPool_CAttendByAdd_outReLU_FLOPs':
                self.channel_interacter = ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_FLOPs(self)
            elif args.interacter=='PWI2_STPool_CAttendByAdd_outReLU_reparam':
                self.channel_interacter = ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_reparam(self)
            elif args.interacter=='PWI2_STPool_CAttendByAdd_outReLU_Impl2':
                self.channel_interacter = ada_full_PWInteraction2_V2_STPool_CAttendByAdd_outReLU_Impl2(self)
            elif args.interacter=='PWI2_CAttendByAdd_outReLU_woCache':
                self.channel_interacter = ada_full_PWInteraction2_V2_CAttendByAdd_outReLU_woCache(self)

            elif args.interacter=='PWI2_CAttendByAdd_outReLU_bn':
                self.channel_interacter = ada_full_PWInteraction2_V2_CAttendByAdd_outReLU_bn(self)
            elif args.interacter=='PWI2_CAttendByAdd_outReLU_MLP':
                self.channel_interacter = ada_full_PWInteraction2_V2_CAttendByAdd_outReLU_MLP(self)
            elif args.interacter=='STPool_CAttendByAddReLU':
                self.channel_interacter = STPool_CAttendByAddReLU(self)
            elif args.interacter=='ada_shift':
                self.channel_interacter = ada_full_shifter(self)
            elif args.interacter=='donothing':
                self.channel_interacter = donothing()
            else:
                raise NotImplementedError
            #self.channel_interacter = full_shifter()
            #self.channel_interacter = ada_full(self)
            #self.channel_interacter = ada_full_PWInteraction(self)            
            #self.channel_interacter = ada_full_shifter_babies(self)
        #*************************************************************************************************
    def forward(self, x):
        #set_trace()
        if x.dim()==4:
            bt, c, h, w = x.size()
            t = self.t
        elif x.dim()==5:
            b, c, t, h, w = x.size()
        else:
            raise ValueError('dim={}'.format(x.dim()))
            
        for i in range(len(self.out_folds)):
            #set_trace()
            if i==0 or (self.insert_propotion=='part' and self.stride!=1):
                sp = x
            else:
                # *************************Channel Interact*************************
                if self.stride!=1 and self.insert_propotion=='all':
                    # upsample
                    if x.size(-1)%1==0 or x.size(-2)%1==0:
                        # to be compatible with odd dimension
                        self.upsample = nn.Upsample(size=(x.size(-2),x.size(-1)), mode='nearest')
                    if out_recursive.dim()==4:
                        upsampled = self.upsample(out_recursive)
                    elif out_recursive.dim()==5:
                        reshaped = out_recursive.permute(0, 2, 1, 3, 4).reshape(-1,out_recursive.size()[1],*out_recursive.size()[-2:])
                        # size(bs*t, c, h, w)
                        upsampled = self.upsample(reshaped)
                        upsampled = upsampled.reshape(-1, t, *upsampled.size()[1:]).permute(0, 2, 1, 3, 4)
                        # size(bs, c, t, h, w)
                    else:
                        raise ValueError('Tensor dim: {}'.format(out_recursive.dim()))
                    '''
                    if upsampled.size(-2)==16:
                        set_trace()
                    '''
                    sp = self.channel_interacter(upsampled, sp, x)
                else:
                    sp = self.channel_interacter(out_recursive, sp, x)

            
            # *************************Convolution*************************
            if type(self.layer[i]) == Conv2d_bn_relu:
                sp = self.layer[i](sp)
            elif type(self.layer[i]) == Conv3d_bn_relu:
                #set_trace()
                if x.dim()==4:
                    sp = self.layer[i](sp.view(bt//t, t, c, h, w).contiguous().permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
                    sp = sp.contiguous().view((bt, self.out_folds[i]) + sp.size()[-2:])
                elif x.dim()==5:
                    #set_trace()
                    sp = self.layer[i](sp)
                else:
                    raise ValueError('dim={}'.format(x.dim()))
            elif type(self.layer[i]) == Conv1d_bn_relu:
                raise NotImplementedError
                sp = self.layer[i](sp.view(bt//t, t, c, h, w).contiguous().permute(0, 3, 4, 2, 1).contiguous().view(bt//t*h*w, c, t)).view(bt//t, h, w, self.out_folds[i], t).contiguous().permute(0, 4, 3, 1, 2).contiguous().view(bt, self.out_folds[i], h, w)

            # *************************store output*************************
            if i==0:
                out = sp
                if 'SepBN' in self.interacter or 'reparam' in self.interacter:
                    raw = self.hook_store[0]
                    if raw.dim()!=4:
                        out_recursive = raw.permute(0, 2, 1, 3, 4).contiguous().view((bt, self.out_folds[i]) + sp.size()[-2:])
                    else:
                        out_recursive = raw
                else:
                    out_recursive = out
            else:
                out = torch.cat((out, sp), 1)
                #set_trace()
                if 'SepBN' in self.interacter or 'reparam' in self.interacter:
                    raw = self.hook_store[0]
                    if raw.dim()!=4:
                        raw_ = raw.permute(0, 2, 1, 3, 4).contiguous().view((bt, self.out_folds[i]) + sp.size()[-2:])
                        out_recursive = torch.cat((out_recursive, raw_), 1)
                    else:
                        out_recursive = torch.cat((out_recursive, raw), 1)
                else:
                    out_recursive = out
        if hasattr(self, 'channel_interacter') and hasattr(self.channel_interacter, 'start_ind'):
            self.channel_interacter.start_ind = 0
        if hasattr(self.channel_interacter, 'net') and hasattr(self.channel_interacter.net, 'start_ind'):
            self.channel_interacter.net.start_ind = 0
        #set_trace()
        return self.affiner(out)
