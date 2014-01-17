#/usr/bin/python2.7
#*-coding:utf-8-*-
from structure import DependencyTree
from DepNN import DepNN

#Main class to execute the task.
class Execute(object):
    def __init__(self):
        self.depnn = DepNN()
        self.depnn.extendLookUp('../data/depcheck2')
        self.depnn.load_wordvector('../data/word-vec.bin')
        self.deptree = DependencyTree()

    def build_model(self):
        self.depnn.build_AutoEncoder(200, '../data/compilation.bin')

    def load_model(self):
        self.depnn.load_AutoEncoder('../data/compilation.bin')


    def train_sentence(self, lines):
        w = self.deptree.read_sent(lines[0])
        y = self.deptree.read_dependency(lines[1])
        #self.depnn.saveRAE('../data/weights.bin')
        #self.depnn.loadRAE('../data/weights.bin')
        print ' '.join(x[0] for x in w)
        
        self.depnn.build_DepRNN_Tree(y)
    
    def dump_weights(self, db_file):
        self.depnn.saveRAE(db_file)
    
    def load_weights(self, db_file):
        self.depnn.loadRAE(db_file)


