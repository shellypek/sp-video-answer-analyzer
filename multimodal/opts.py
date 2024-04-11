# -*- coding: utf-8 -*-
'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

class Opts:
    def __init__(self):
        # Default values
        self.annotation_path = 'annotations.txt'
        self.result_path = 'results'
        self.store_name = 'model'
        self.dataset = 'RAVDESS'
        self.n_classes = 8
        self.model = 'multimodalcnn'
        self.num_heads = 1
        self.device = 'cpu'
        self.sample_size = 224
        self.sample_duration = 15
        self.learning_rate = 0.04
        self.momentum = 0.9
        self.lr_steps = [40, 55, 65, 70, 200, 250]
        self.dampening = 0.9
        self.weight_decay = 1e-3
        self.lr_patience = 10
        self.batch_size = 8
        self.n_epochs = 10
        self.begin_epoch = 1
        self.resume_path = ''
        self.pretrain_path = 'EfficientFace_Trained_on_AffectNet7.pth.tar'
        self.no_train = False
        self.no_val = False
        self.test = True
        self.test_subset = 'test'
        self.n_threads = 16
        self.video_norm_value = 255
        self.manual_seed = 1
        self.fusion = 'ia'
        self.mask = 'softhard'

# Instantiate the options object
opts = Opts()
