from __future__ import division
from matplotlib import pyplot as plt
import os
import numpy as np
import PIL.Image as pil
import tensorflow as tf
from SfMLearner import SfMLearner
from utils import normalize_depth_for_display


def main():
    img_height=128
    img_width=416
    ckpt_file = 'models/model-190532'
    fh = open('misc/sample.png', 'r')
    I = pil.open(fh)
    I = I.resize((img_width, img_height), pil.ANTIALIAS)
    I = np.array(I)

    sfm = SfMLearner()
    sfm.setup_inference(img_height,
                        img_width,
                        mode='depth')

    saver = tf.train.Saver([var for var in tf.model_variables()]) 
    with tf.Session() as sess:
        saver.restore(sess, ckpt_file)
        pred = sfm.inference(I[None,:,:,:], sess, mode='depth')

    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1); plt.imshow(I)
    plt.subplot(1,2,2); plt.imshow(normalize_depth_for_display(pred['depth'][0,:,:,0]))
    # imgs = ['100','030','250','400']
    # count=0
    # sfm = SfMLearner()
    # sfm.setup_inference(img_height,
    #                     img_width,
    #                     mode='depth')
    # saver = tf.train.Saver([var for var in tf.model_variables()]) 


    # for j in imgs:
    #     fh = open('/home/skotasai/SfMLearner/KITTI/2011_09_26/2011_09_26_drive_0093_sync/image_02/data/0000000'+j+'.png', 'r')
    #     I = pil.open(fh)
    #     I = I.resize((img_width, img_height), pil.ANTIALIAS)
    #     I = np.array(I)
if __name__ == '__main__':
    main()
