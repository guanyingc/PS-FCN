from .base_opts import BaseOpts
class RunModelOpts(BaseOpts):
    def __init__(self):
        super(RunModelOpts, self).__init__()
        self.initialize()

    def initialize(self):
        BaseOpts.initialize(self)
        #### Model and Dataset ####
        self.parser.add_argument('--run_model',   default=True, action='store_false')
        self.parser.add_argument('--benchmark',   default='DiLiGenT_main')
        self.parser.add_argument('--bm_dir',      default='data/datasets/DiLiGenT/pmsData')
        self.parser.add_argument('--model',       default='PS_FCN_run')
        self.parser.add_argument('--test_batch',  default=1,    type=int)

        #### Display Arguments ####
        self.parser.add_argument('--test_intv',   default=1,    type=int) 
        self.parser.add_argument('--test_disp',   default=1,    type=int) 
        self.parser.add_argument('--test_save',   default=1,    type=int) 

    def parse(self):
        BaseOpts.parse(self)
        return self.args
