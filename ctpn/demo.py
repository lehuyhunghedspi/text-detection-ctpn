from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg


def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img,image_name,boxes,scale):
    base_name = image_name.split('/')[-1]
    with open('/content/drive/My Drive/GR2/ctpn/finetune/results/' + '{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
            max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))

            line = ','.join([str(min_x),str(min_y),
                            str(max_x),str(min_y),
                            str(max_x),str(max_y),
                            str(min_x),str(max_y)])+'\r\n'
            f.write(line)

    img=cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("/content/drive/My Drive/GR2/ctpn/finetune/results_visualize/", base_name), img)

def ctpn(sess, net, image_name):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes,mask = test_ctpn(sess, net, img)

    mask_class=np.argmax(mask,axis=-1)[0]

    print(np.unique(mask_class))
    print(mask_class.shape,mask_class[0,1])
    mask_class[mask_class == 1] = 125

    mask_class[mask_class == 2] = 255
    print(np.unique(mask_class))
    print(mask_class.shape)
    cv2.imwrite("/content/debug/mask_"+os.path.basename(image_name),mask_class)
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes(img, image_name, boxes, scale)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))



if __name__ == '__main__':
    if os.path.exists("/content/drive/My Drive/GR2/ctpn/results/"):
        shutil.rmtree("/content/drive/My Drive/GR2/ctpn/results/")
        print("====== remove result folder =======")
    os.makedirs("/content/drive/My Drive/GR2/ctpn/results/")
    if os.path.exists("/content/drive/My Drive/GR2/ctpn/results_visualize/"):
        shutil.rmtree("/content/drive/My Drive/GR2/ctpn/results_visualize/")
        print("====== remove result folder results_visualize=======")
    os.makedirs("/content/drive/My Drive/GR2/ctpn/results_visualize/")

    cfg_from_file('/content/text-detection-ctpn/ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        print('==============')
        print(cfg.TEST.checkpoints_path)
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print(ckpt)
        # print('Restoring from {}...'.format("/content/text-detection-ctpn/checkpoints/VGGnet_fast_rcnn_iter_10500.ckpt"), end=' ')
        saver.restore(sess,"/content/text-detection-ctpn/checkpoints/VGGnet_fast_rcnn_iter_8000.ckpt" )
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _,_ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'test', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'test', '*.jpg'))

    for im_name_count,im_name in enumerate(im_names):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(im_name_count,len(im_names),('Demo for {:s}'.format(im_name)))
        ctpn(sess, net, im_name)
        if im_name_count==5:
            break

