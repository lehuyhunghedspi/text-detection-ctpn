# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from .network import Network
from ..fast_rcnn.config import cfg

class VGGnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.gt_ishard = tf.placeholder(tf.int32, shape=[None], name='gt_ishard')
        self.dontcare_areas = tf.placeholder(tf.float32, shape=[None, 4], name='dontcare_areas')
        self.keep_prob = tf.placeholder(tf.float32)
        self.mask_label=tf.placeholder(tf.int32, shape=[None,None,None], name='mask_label')
        
        self.layers = dict({'data':self.data, 
                            'im_info':self.im_info,
                            'gt_boxes':self.gt_boxes,\
                            'gt_ishard': self.gt_ishard, 
                            'dontcare_areas': self.dontcare_areas,
                            'mask_label':self.mask_label})
        self.trainable = trainable

        self.setup()

    def setup(self):

        # n_classes = 21
        n_classes = cfg.NCLASSES
        # anchor_scales = [8, 16, 32]
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [16, ]

        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3'))
        #========= RPN ============
        (self.feed('conv5_3')
             .conv(3,3,512,1,1,name='rpn_conv/3x3'))

        (self.feed('rpn_conv/3x3').Bilstm(512,128,512,name='lstm_o'))
        (self.feed('lstm_o').lstm_fc(512,len(anchor_scales) * 10 * 4, name='rpn_bbox_pred'))
        (self.feed('lstm_o').lstm_fc(512,len(anchor_scales) * 10 * 2,name='rpn_cls_score'))

        # generating training labels on the fly
        # output: rpn_labels(HxWxA, 2) rpn_bbox_targets(HxWxA, 4) rpn_bbox_inside_weights rpn_bbox_outside_weights
        # 给每个anchor上标签，并计算真值（也是delta的形式），以及内部权重和外部权重
        (self.feed('rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info')
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' ))

        # shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        # 给之前得到的score进行softmax，得到0-1之间的得分
        (self.feed('rpn_cls_score')
             .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape')
             .spatial_softmax(name='rpn_cls_prob'))

        print(self.layers.keys())   


        # define mask unet
        


        (self.feed('conv5_3')
             .transpose_conv(3,3,512,512,2,2,name='transpose_pool5'))

        (self.feed('conv4_3','transpose_pool5')
             .crop_and_concat(name='concat_pool4'))

        (self.feed('concat_pool4')
            .conv(3, 3, 512, 1, 1,c_i=1024, name='concat_pool4_c1'))
        (self.feed('concat_pool4_c1')
            .conv(3, 3, 512, 1, 1,c_i=512, name='concat_pool4_c2'))

        (self.feed('concat_pool4_c2')
             .transpose_conv(3,3,512,256,2,2,name='transpose_pool4'))

        (self.feed('conv3_3','transpose_pool4')
             .crop_and_concat(name='concat_pool3'))


        (self.feed('concat_pool3')#x,x,512
            .conv(3, 3, 256, 1, 1,c_i=512, name='concat_pool3_c1'))
        (self.feed('concat_pool3_c1')
            .conv(3, 3, 256, 1, 1,c_i=256, name='concat_pool3_c2'))
        (self.feed('concat_pool3_c2')#x,x,256
             .transpose_conv(3,3,256,128,2,2,name='transpose_pool3'))

        (self.feed('conv2_2','transpose_pool3')
             .crop_and_concat(name='concat_pool2'))


        (self.feed('concat_pool2')#x,x,256
            .conv(3, 3, 128, 1, 1,c_i=256, name='concat_pool2_c1'))
        (self.feed('concat_pool2_c1')
            .conv(3, 3, 128, 1, 1,c_i=128, name='concat_pool2_c2'))
        (self.feed('concat_pool2_c2')#x,x,128
             .transpose_conv(3,3,128,64,2,2,name='transpose_pool2'))

        (self.feed('conv1_2','transpose_pool2')#x,x,128
             .crop_and_concat(name='concat_pool1'))

        (self.feed('concat_pool1')#x,x,128
             .conv(3, 3, 64, 1, 1,c_i=128, name='concat_pool1_c1'))

        (self.feed('concat_pool1_c1')#x,x,64
             .conv(3, 3, 64, 1, 1,c_i=64, name='concat_pool1_c2'))

        (self.feed('concat_pool1_c2')#x,x,64
             .conv(3, 3, 3, 1, 1,c_i=64, name='logit_mask',relu=False))



        # exit(-1)
