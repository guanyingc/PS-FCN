import argparse

class BaseOpts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        #### Device Arguments ####
        self.parser.add_argument('--cuda',        default=True,  action='store_false')
        self.parser.add_argument('--time_sync',   default=False, action='store_true')
        self.parser.add_argument('--workers',     default=8,     type=int)
        self.parser.add_argument('--seed',        default=0,     type=int)

        #### Model Arguments ####
        self.parser.add_argument('--fuse_type',  default='max')
        self.parser.add_argument('--in_light',   default=True,  action='store_false')
        self.parser.add_argument('--use_BN',     default=False, action='store_true')
        self.parser.add_argument('--in_img_num', default=32,    type=int)
        self.parser.add_argument('--start_epoch',default=1,     type=int)
        self.parser.add_argument('--epochs',     default=30,    type=int)
        self.parser.add_argument('--resume',     default=None)
        self.parser.add_argument('--retrain',    default=None)

        #### Log Arguments ####
        self.parser.add_argument('--save_root', default='data/Training/') 
        self.parser.add_argument('--item',      default='calib')

    def parse(self):
        self.args = self.parser.parse_args()
        return self.args
