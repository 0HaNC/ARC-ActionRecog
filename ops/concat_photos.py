import numpy as np
import os.path as osp
import os
import cv2
from ipdb import set_trace

def cat_photos(folder, test_list, tmpl= '{:05d}.jpg', plot=False):
    ret = None
    frms = []
    for ind in test_list:
        frm_pth = osp.join(folder, tmpl.format(ind))
        frm = cv2.imread(frm_pth)
        if ret is None:
            ret = np.zeros_like(frm)
        frms.append(frm)
    
    cated = np.concatenate(frms, axis=0).astype(np.int)
    print(cated.shape)
    if plot:
        cv2.imwrite('cated_plot.jpg', cated)
    return cated

if __name__=='__main__':
    folder = '/home/linhanxi/smth-v1/20bn-something-something-v1/35089/'
    test_list = [10,11]
    cat_photos(folder, test_list, tmpl= '{:05d}.jpg', plot=True)
