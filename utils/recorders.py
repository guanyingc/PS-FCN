from collections import OrderedDict
import numpy as np

class Records(object):
    """
    Records->Train,Val>Loss,Accuracy->Epoch1,2,3->[v1,v2]
    IterRecords->Train,Val->Loss, Accuracy,->[v1,v2]
    """
    def __init__(self, log_dir, records=None):
        if records == None:
            self.records = OrderedDict()
        else:
            self.records = records
        self.iter_rec = OrderedDict()
        self.log_dir  = log_dir

    def resetIter(self):
        self.iter_rec.clear()

    def checkDict(self, a_dict, key, sub_type='dict'):
        if key not in a_dict.keys():
            if sub_type == 'dict':
                a_dict[key] = OrderedDict()
            if sub_type == 'list':
                a_dict[key] = []

    def updateIter(self, split, keys, values):
        self.checkDict(self.iter_rec, split, 'dict')
        for k, v in zip(keys, values):
            self.checkDict(self.iter_rec[split], k, 'list')
            self.iter_rec[split][k].append(v)

    def saveIterRecord(self, epoch, reset=True):
        for s in self.iter_rec.keys(): # split
            self.checkDict(self.records, s, 'dict')
            for k in self.iter_rec[s].keys():
                self.checkDict(self.records[s], k, 'dict')
                self.checkDict(self.records[s][k], epoch, 'list')
                self.records[s][k][epoch].append(np.mean(self.iter_rec[s][k]))
        if reset: 
            self.resetIter()

    def insertRecord(self, split, key, epoch, value):
        self.checkDict(self.records, split, 'dict')
        self.checkDict(self.records[split], key, 'dict')
        self.checkDict(self.records[split][key], epoch, 'list')
        self.records[split][key][epoch].append(value)

    def iterRecToString(self, split, epoch):
        classes = ['loss', 'acc', 'err']
        rec_strs = ''
        for c in classes:
            strs = ''
            for k in self.iter_rec[split].keys():
                if (c in k.lower()):
                    strs += '{}: {:.3f}| '.format(k, np.mean(self.iter_rec[split][k]))
            if strs != '':
                rec_strs += '\t [{}] {}\n'.format(c.upper(), strs)
        self.saveIterRecord(epoch)
        return rec_strs

    def epochRecToString(self, split, epoch):
        classes = ['loss', 'acc', 'err']
        rec_strs = ''
        for c in classes:
            strs = ''
            for k in self.records[split].keys():
                if (c in k.lower()) and (epoch in self.records[split][k].keys()):
                    strs += '{}: {:.3f}| '.format(k, np.mean(self.records[split][k][epoch]))
            if strs != '':
                rec_strs += '\t [{}] {}\n'.format(c.upper(), strs)
        return rec_strs
