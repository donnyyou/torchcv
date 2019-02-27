import torch
import numpy as np
from model.dmnetU import l_Net
import model.ICNet_model as ICNet_model
import cv2
import os

from utils.helpers.video_helper import VideoReader
from utils.tools.configer import Configer


class PersonSegmentor(object):

    def __init__(self, configer):
        self.configer = configer
        self.device = torch.device('cuda' if self.configer.exists('gpu_id') else 'cpu')
        self._load_model()

    def _load_model(self):
        self.fabby_net = ICNet_model.ICNet(32)
        self.fabby_net.load_state_dict(torch.load(self.configer.get('fabby_model'),
                                       map_location=lambda storage, loc: storage))
        self.fabby_net.to(self.device)
        self.fabby_net.eval()
        # trimap net
        self.trimap_net = ICNet_model.TrimapNet(64)
        self.trimap_net.load_state_dict(torch.load(self.configer.get('trimap_model'),
                                        map_location=lambda storage, loc: storage))
        self.trimap_net.to(self.device)
        self.trimap_net.eval()

        # matting net
        self.mat_net = l_Net(16, 64, dmTpye='small')
        self.mat_net.load_state_dict(torch.load(self.configer.get('mat_model'),
                                   map_location=lambda storage, loc: storage))
        self.mat_net.to(self.device)
        self.mat_net.eval()

    def forward(self, image):
        imgs, img = self._get_in_data3c(image, self.configer.get('img_size'))
        imgs = imgs.to(self.device)
        img = img.to(self.device)
        fm, seg_out = self.fabby_net(imgs, img)
        seg_out = torch.sigmoid(seg_out)
        trimap = self.trimap_net(fm.detach())  # don't train segmentation net
        imgs = imgs * (1. / 255)  # img is 0~255
        img = torch.cat([imgs, seg_out[:, 1:2].detach(), trimap], 1)
        outputs, out_refine = self.mat_net(img)

        out_refine = torch.clamp(out_refine, 0, 1)
        out_refine = out_refine.cpu().data[0][0].numpy()
        return out_refine

    @staticmethod
    def _get_in_data3c(image, img_size):
        # im = Image.open(img_file)
        in_ = np.array(image, dtype=np.float32)
        # in_ = in_[:, :, ::-1]
        in_ = cv2.resize(in_, img_size, interpolation=cv2.INTER_CUBIC)
        in_2 = cv2.resize(in_, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        in_ = in_.transpose((2, 0, 1))
        in_2 = in_2.transpose((2, 0, 1))
        img = torch.from_numpy(in_[np.newaxis])
        img_2 = torch.from_numpy(in_2[np.newaxis])
        return img_2, img

    @staticmethod
    def parse(image, mask, out_size=(240, 320)):
        image_canvas = cv2.resize(image, out_size, interpolation=cv2.INTER_CUBIC)
        mask_canvas = cv2.resize(mask, out_size, interpolation=cv2.INTER_CUBIC)
        mask_canvas = np.repeat((mask_canvas * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
        # print(mask_canvas.shape)
        image_canvas = cv2.addWeighted(image_canvas, 0.6, mask_canvas, 0.8, 0)
        return image_canvas

if __name__ == '__main__':
    # segmention net init
    CONFIG_DICT = {
        'img_size': (480, 640),
        'gpu_id': [0],
        'fabby_model': './model/ICNet32Photo640x480_iter_310000.pth',
        'trimap_model': './model/trimap_icnet32SimpleV2_iter_120000.pth',
        'mat_model': './model/trimap_icnet32SimpleV2_matting_iter_120000.pth'
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in CONFIG_DICT['gpu_id'])

    person_segmentor = PersonSegmentor(Configer(config_dict=CONFIG_DICT))
    image = cv2.imread('./samples/25962783011.jpg')
    print(image.shape)
    out = person_segmentor.forward(image)
    cv2.imshow("main", person_segmentor.parse(image, out))
    cv2.waitKey()
    video_reader = VideoReader('./samples/24866_25155.avi')
    for image in video_reader:
        print(image.shape)
        out = person_segmentor.forward(image)
        cv2.imshow("main", person_segmentor.parse(image, out))
        #cv2.imshow("main", frame)
        cv2.waitKey()

