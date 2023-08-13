# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
from functools import partial
from ipdb import set_trace
ROOT_DATASET = '/ssd/video/'  # '/data/jilin/'
#ROOT_DATASET = '/media/sda/data/'
# list root
ROOT_DATASET = '/home/linhanxi/'

def return_diving48_V2(modality):
    filename_categories = 'Diving48/category.txt'
    if modality == 'RGB':    
        root_data = '/home/linhanxi/diving48_240p/frames_240p/'
        filename_imglist_train = '/home/linhanxi/diving48_240p/V2/train_videofolder.txt'
        filename_imglist_val = '/home/linhanxi/diving48_240p/V2/val_videofolder.txt'
        prefix = 'frames{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)    
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_ucf101(modality):
    filename_categories = 'UCF101/labels/classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_rgb_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_flow_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality, name):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51/'
        if 'sp1' in name:
            filename_imglist_train = 'HMDB51/train_videofolder_split1.txt'
            filename_imglist_val = 'HMDB51/val_videofolder_split1.txt'
        elif 'sp2' in name:
            filename_imglist_train = 'HMDB51/train_videofolder_split2.txt'
            filename_imglist_val = 'HMDB51/val_videofolder_split2.txt'
        elif 'sp3' in name:
            filename_imglist_train = 'HMDB51/train_videofolder_split3.txt'
            filename_imglist_val = 'HMDB51/val_videofolder_split3.txt'
        else:
            raise ValueError
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 'github/tsm.backup/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'smth-v1/20bn-something-something-v1'
        filename_imglist_train = 'tsm.backup/train_videofolder.txt'
        filename_imglist_val = 'tsm.backup/val_videofolder.txt'
        #filename_imglist_val = 'sth-v1/labels/train_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-flow'
        filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    set_trace()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 'something/v2/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-frames'
        filename_imglist_train = 'something/v2/train_videofolder.txt'
        filename_imglist_val = 'something/v2/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ROOT_DATASET + 'jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_diving48(modality):
    filename_categories = 'Diving48/category.txt'
    if modality == 'RGB':    
        root_data = ROOT_DATASET + 'Diving48/frames'
        filename_imglist_train = 'Diving48/train_videofolder.txt'
        filename_imglist_val = 'Diving48/val_videofolder.txt'
        prefix = 'frames{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)    
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        #root_data = '/media/nvme1n1p1/linhanxi/k400'
        #filename_imglist_train = 'nju_list/train2.csv'
        #filename_imglist_val = 'official_list/val2.txt'
        root_data = ''
        filename_imglist_train = '/home/linhanxi/k400/3parts_list/train_new.txt'
        filename_imglist_val = '/home/linhanxi/k400/3parts_list/val.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_kinetics_online(modality):
    filename_categories = 400
    if modality == 'RGB':
        # data root
        root_data = '/home/linhanxi/k400_online'
        filename_imglist_train = 'k400_online/val_online.txt'
        # filename_imglist_train = 'k400_online/train_online.csv'
        filename_imglist_val = 'k400_online/val_online.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    global ROOT_DATASET
    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51_sp1': partial(return_hmdb51, name='sp1'), 'hmdb51_sp2': partial(return_hmdb51, name='sp2'),
                    'hmdb51_sp3': partial(return_hmdb51, name='sp3'), 'diving48':return_diving48,
                   'kinetics': return_kinetics, 'kinetics_online': return_kinetics_online,
                   'diving48_V2':return_diving48_V2,}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)
    if dataset == 'kinetics':
        ROOT_DATASET = root_data
    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
