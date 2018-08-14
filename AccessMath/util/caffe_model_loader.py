import caffe

from AccessMath.preprocessing.config.parameters import Parameters

class CaffeModelLoader(object):
    def __init__(self,
                 model_def=Parameters.Model_TextDetection+'deploy.prototxt',
                 model_weights=Parameters.Model_TextDetection+'v2.2_VGG_text_longer_conv_300x300_iter_10000.caffemodel'):
        caffe.set_mode_gpu()
        caffe.set_device(Parameters.GPU_TextDetection)
        print('Loading {} {}'.format(model_def, model_weights))
        self.net = caffe.Net(model_def,  # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)  # use test mode (e.g., don't perform dropout)
        print('Loading complete')

    def getModel(self):
        return self.net
