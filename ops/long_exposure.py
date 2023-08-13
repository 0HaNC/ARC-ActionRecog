import numpy as np
import os.path as osp
import os
import cv2
from ipdb import set_trace

def exposure(folder, start_ind, span, weights, tmpl= '{:05d}.jpg', plot=False):
    ret = None
    frms = []
    for ind in range(start_ind, start_ind+span):
        frm_pth = osp.join(folder, tmpl.format(ind))
        frm = cv2.imread(frm_pth)
        if ret is None:
            ret = np.zeros_like(frm)
        frms.append(frm)
    
    weights =  np.exp(weights) / sum(np.exp(weights))
    weights=weights[:, np.newaxis, np.newaxis, np.newaxis]
    print(weights)
    frms = np.array(frms)
    exposured = (weights*frms).sum(axis=0).astype(np.int)
    if plot:
        cv2.imwrite('exposured_plot.jpg', exposured)
    return exposure

if __name__=='__main__':
    folder = '/home/linhanxi/smth-v1/20bn-something-something-v1/35089/'
    test_span = 3
    _ = exposure(folder, 10, test_span, [20]*test_span, tmpl= '{:05d}.jpg', plot=True)
