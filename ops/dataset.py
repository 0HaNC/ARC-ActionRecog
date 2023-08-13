# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path as osp
import numpy as np
from numpy.random import randint
from ipdb import set_trace
# from .utils import time_counter
import skvideo.io as io
import skvideo
import cv2
from .decode_on_the_fly import _load_action_frame_nums_to_4darray

def get_vid_meta(vid_path, keys):
    # typical usage: get_vid_meta(vid_path, ['@nb_frames', '@width', '@height'])
    meta = io.ffprobe(vid_path)['video']
    ret = {}
    for k in keys:
        try:
            ret[k] = meta[k]
        except:
            set_trace()
    return ret

def get_vid_thw(vid_path):
    # typical usage: get_vid_thw(vid_path)
    vid_reader = skvideo.io.FFmpegReader(vid_path)
    t,h,w,_ = vid_reader.getShape()
    # important for avoid mem leak
    vid_reader.close()
    meta = {}
    meta['length'], meta['height'], meta['width']  = t, h, w
    return meta

def video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        #print("could not open: ", video_path)
        return -1
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    return length, width, height

def return_file(vid_path):
    with open(vid_path, 'rb') as f:
        vid_file = f.read()
    return vid_file

# vid_path, indices =, [1, 10, 30, 35]
# with open(vid_path, 'rb') as f:
#   vid_file = f.read()
# meta = get_vid_meta(vid_path,  ['@nb_frames', '@width', '@height'])
# decoded_frames = _load_action_frame_nums_to_4darray(vid_file, frame_nums=indices,  width=int(meta.get('@width')), height=int(meta.get('@height'))))
# (T,H,W,C)
# process_data = np.asarray(decoded_frames, dtype=np.float32)

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, dataset, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False, 
                 scale=1, online_decode=False, args=None):
        self.dataset = dataset
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        self.scale = scale
        self.online_decode = online_decode
        self.args = args
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if 'Slow' in self.args.arch:
            # default 8x8
            self.fast_frms = 8 * self.num_segments
            # alpha is 8 by default
            self.fast_stride = self.args.slow_stride//8
            # slow_stride is 8 by default
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        interval = int(1 // self.scale)
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        #import pdb;pdb.set_trace()
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]
        #self.video_list = self.video_list[:int(len(self.video_list) * self.scale)]
        self.video_list = self.video_list[::interval]
        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            if 'Slow' not in self.args.arch:
                sample_pos = max(1, 1 + record.num_frames - 64)
                t_stride = 64 // self.num_segments
                start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
                offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            else:
                sample_pos = max(1, 1 + record.num_frames - self.fast_frms)
                start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
                offsets = [(idx * self.fast_stride + start_idx) % record.num_frames for idx in range(self.fast_frms)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            if 'Slow' not in self.args.arch:
                sample_pos = max(1, 1 + record.num_frames - 64)
                t_stride = 64 // self.num_segments
                start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
                offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            else:
                sample_pos = max(1, 1 + record.num_frames - self.fast_frms)
                start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
                offsets = [(idx * self.fast_stride + start_idx) % record.num_frames for idx in range(self.fast_frms)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            if 'Slow' not in self.args.arch:
                sample_pos = max(1, 1 + record.num_frames - 64)
                t_stride = 64 // self.num_segments
                start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
                offsets = []
                for start_idx in start_list.tolist():
                    offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            else:
                sample_pos = max(1, 1 + record.num_frames - self.fast_frms)
                start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
                offsets = []
                for start_idx in start_list.tolist():
                    offsets += [(idx * self.fast_stride + start_idx) % record.num_frames for idx in range(self.fast_frms)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        if not self.online_decode:
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)
        else:
            full_path = os.path.join(self.root_path, record.path)
            if osp.splitext(record.path) == '':
                record.path += '.mp4'
            
        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)
        # online_read and count frms
        meta = {}
        if self.online_decode:
            '''
            # skvideo.io
            readed_ndarr = io.vread(osp.join(self.root_path, record.path))
            record._data[1] = len(readed_ndarr)
            '''
            '''
            # io.ffprobe
            meta = get_vid_meta(full_path,  ['@nb_frames', '@width', '@height'])
            record._data[1] = meta.get('@nb_frames')
            '''
            '''
            # cv2, cannot provide accurate num_frms..
            length, width, height = video_frame_count(full_path)
            meta['length'], meta['width'], meta['height']  = length, width, height
            record._data[1] = length
            '''
            # skvideo.io.FFmpegReader
            meta = get_vid_thw(full_path)
            record._data[1] =  meta['length']-1 # cope with the issue(https://github.com/dukebw/lintel/issues/31#issuecomment-714418804)
            meta['full_path'] = full_path

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            if self.dataset == 'kinetics':
                # for debug
                # set_trace()
                #segment_indices = self._get_val_indices(record)
                segment_indices = self._sample_indices(record)
            else:
                segment_indices = self._get_test_indices(record)
        # set_trace()
        return self.get(record, segment_indices, meta)

    def get(self, record, indices, meta=None):
        if not self.online_decode:
            images = list()
            for seg_ind in indices:
                p = int(seg_ind)
                for i in range(self.new_length):
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += 1
        else:
            # print(len(readed_ndarr), record.num_frames)
            try:
                #skvideo.io.vread
                #images = readed_ndarr[indices-1]
                vid_file = return_file(meta.get('full_path'))
                # lintel.vread io index start from 0.
                # And also note that Lintel seems to miss the first frame of a video(https://github.com/dukebw/lintel/issues/31#issuecomment-714418804)
                images = _load_action_frame_nums_to_4darray(vid_file, frame_nums=indices, width=int(meta.get('width')), height=int(meta.get('height')))
                del vid_file
                #print(meta.get('full_path'), indices[-1], record.num_frames)
                #images = np.asarray(decoded_frames, dtype=np.float32)
            except:
                set_trace()
        '''
        process_data = self.transform(images)
        return process_data, record.label
        '''
        #set_trace()
        if self.transform is not None:
            process_data,record_label = self.transform((images,record.label))
            return process_data,record_label
        else:
            return torch.tensor([np.array(img) for img in images]), record.label


    def __len__(self):
        return len(self.video_list)
