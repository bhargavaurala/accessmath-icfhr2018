import torch

from AccessMath.preprocessing.config.parameters import Parameters

class TorchModelLoader(object):
    def __init__(self,
                 net, # Pass torch.nn.module object here with a chosen structure
                 model_path=Parameters.Model_PersonDetection+'ssd_300_VOC0712.pth',
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