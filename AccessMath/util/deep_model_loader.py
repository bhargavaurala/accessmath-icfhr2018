import os

import torch
import caffe

caffe_root = '/home/buralako/git/TextBoxes'
caffe.set_device(0)
caffe.set_mode_gpu()

class CaffeModelLoader(object):
    def __init__(self,
                 model_def=os.path.join(caffe_root, 'examples/TextBoxes/deploy.prototxt'),
                 model_weights=os.path.join(caffe_root, 'examples/TextBoxes/TextBoxes_icdar13.caffemodel')):
        print('Loading {} {}'.format(model_def, model_weights))
        self.net = caffe.Net(model_def,  # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)  # use test mode (e.g., don't perform dropout)
        print('Loading complete')

    def getModel(self):
        return self.net

class TorchModelLoader(object):
    def __init__(self,
                 net, # Pass torch.nn.module object here with a chosen structure
                 model_path='/home/buralako/git/ssd.pytorch/weights/ssd_300_VOC0712.pth',
                 cuda=True):
        self.net = net
        self.model_path = model_path
        self.cuda = cuda
        print('Loading {}'.format(model_path))
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()
        if self.cuda:
            self.net.cuda()
        print('Loading complete')

    def getModel(self):
        return self.net

