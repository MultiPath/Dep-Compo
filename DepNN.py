#/usr/bin/python2.7
#*-coding:utf-8-*-
from RAE import RAE, AE
from gensim import models
from copy import deepcopy
import numpy as np
import cPickle as pickle
import theano

#Describe the look-up table of Dep-NN
class DepNN(object):
    def build_DepRNN_Tree(self, table, parent='ROOT', floor=0, flag=0):
        ''' build the recurrent tree for training or predicting'''
        # flag == 0 : training/ flag == 1 : only get the vector.
        
        F = sorted([x for x in table if x[0] == parent], key=lambda p:p[3])
        #print floor, F

        if parent == 'ROOT':
            return self.build_DepRNN_Tree(table, F[0][2], floor+1, flag)


        vec = np.array(self.wordvector[parent], dtype = theano.config.floatX)
        if len(F) == 0:
            return vec
        else:
            R = []
            for fi in F:
                fivec = self.build_DepRNN_Tree(table, fi[2], floor+1, flag)
                #print fi[2], len(fivec)
                #print parent,len(vec)
                depr  = fi[4]
                u = np.concatenate((vec, fivec))
                #print len(u)
                
                d = self.depList(depr)
                v = []
                #print depr,d
                for di in d:
                    if flag == 0:
                        cost, newvec = self.depAE[di].train(u)
                    else:
                        cost, newvec = self.depAE[di].predict(u)
                    v.append(newvec)
                
                R.append(v[0])  #Version .1
            
            RA  = np.concatenate(R)
            num = len(RA)/self.vector_dim

            if num == 1:
                return RA
            else:
                #print 'xxx',len(RA)/200
                if flag == 0:
                    cost, RA = self.depRAE.AEs[num-2].train(RA)
                else:
                    cost, RA = self.depRAE.AEs[num-2].predict(RA)
                
                #print xvec,cost
                return RA
                
    def depList(self, depr):
        r = []
        if depr not in self.depLookUp:
            if depr[0:4] == 'prep':
                depr = 'prep'
            elif depr[0:4] == 'conj':
                depr = 'conj'
            else:
                depr = 'dep'

        while self.depLookUp[depr]:
            r.append(depr)
            depr = self.depLookUp[depr]
        r.append(depr)
        return r


    def build_AutoEncoder(self, n_vector = 200, save_db = None):
        print 'begin.....AE'
        ae  = AE(numpy_rng = self.numpy_rng, n_vector = n_vector)
        print 'beign.....RAE'
        rae = RAE(numpy_rng = self.numpy_rng, n_vector = n_vector)

        print 'deepcopy...'
        for w in self.depLookUp:
            #if w == 'det':
            #print w
            self.depAE[w] = deepcopy(ae)
            pass
        print 'build ok.'
        if save_db != None:
            output = open(save_db, 'wb')
            pickle.dump(self.depAE,  output)
            pickle.dump(self.depRAE, output)
            output.close()
            print 'save ok.'

        self.depRAE = rae


    def load_AutoEncoder(self, load_db):
        input = open(load_db, 'rb') 
        print 'load AE'
        self.depAE = pickle.load(input)
        print 'load RAE'
        self.depRAE= pickle.load(input)
        print 'load ok.'
    


    def load_wordvector(self, bin_file):
        self.wordvector = models.word2vec.Word2Vec.load_word2vec_format(bin_file, binary=True)  
    
    def extendLookUp(self, depCheck, limits = 300):
        deps = [line.split()[0] for line in open(depCheck)if int(line.split()[1]) > limits]
        for d in deps:
            if (d[0:4] == 'prep') and (d != 'prep'):
                self.depLookUp[d] = 'prep'
            if (d[0:4] == 'conj') and (d != 'conj'):
                self.depLookUp[d] = 'conj'
    
    def saveRAE(self, db_file):
        print 'save begin.'
        output = open(db_file, 'wb')
        Wae = {}
        for w in self.depAE:
            Wae[w] = (self.depAE[w].W.get_value(),
                      self.depAE[w].b.get_value(),
                      self.depAE[w].b_dash.get_value())
        Wrae = (self.depRAE.W.get_value(), self.depRAE.b.get_value(), self.depRAE.b_dash.get_value())
        pickle.dump(Wae, output)
        pickle.dump(Wrae, output)
        output.close()
        print 'save ok.'

    def loadRAE(self, db_file):
        print 'load begin.'
        input = open(db_file, 'rb')
        Wae   = pickle.load(input)
        Wrae  = pickle.load(input)
        for w in Wae:
            self.depAE[w].W.set_value(Wae[w][0])
            self.depAE[w].b.set_value(Wae[w][1])
            self.depAE[w].b_dash.set_value(Wae[w][2])

        self.depRAE.W.set_value(Wrae[0])
        self.depRAE.b.set_value(Wrae[1])
        self.depRAE.b_dash.set_value(Wrae[2])
        input.close()
        print 'load ok.'

    def __init__(self):
        self.seed = 100
        self.numpy_rng  = np.random.RandomState(self.seed)
        self.wordvector = None
        self.vector_dim = 200
        self.depRAE = None
        self.depAE  = {}
        self.depLookUp = {
            'plus': None, # The hori-relation #
            'root': None,
            'dep' : None,
                'discourse' : 'dep',
                'aux' : 'dep',
                    'auxpass' : 'aux',
                    'cop'     : 'aux',
                'arg' : 'dep',
                    'agent'   : 'arg',
                    'comp'    : 'arg',
                        'acomp'  : 'comp',
                        'attr'   : 'comp',
                        'ccomp'  : 'comp',
                        'xcomp'  : 'comp',
                        'obj'    : 'comp',
                            'dobj'      : 'obj',
                            'iobj'      : 'obj',
                            'pobj'      : 'obj',
                        'rel'    : 'comp',
                    'subj'    : 'arg',
                        'nsubj'  : 'subj',
                            'nsubjpass' : 'nsubj',
                        'csubj'  : 'subj',
                            'csubjpass' : 'csubj',
                'cc'  : 'dep',
                'conj': 'dep',
                'expl': 'dep',
                'mod' : 'dep',
                    'amod'    : 'mod',
                    'appos'   : 'mod',
                    'advcl'   : 'mod',
                    'det'     : 'mod',
                    'predet'  : 'mod',
                    'infmod'  : 'mod',
                    'mwe'     : 'mod',
                        'mark'   : 'mwe',
                    'partmod' : 'mod',
                    'advmod'  : 'mod',
                        'neg'    : 'advmod',
                    'rcmod'   : 'mod',
                    'quantmod': 'mod',
                    'nn'      : 'mod',
                    'npadvmod': 'mod',
                        'tmod'   : 'npadvmod',
                    'num'     : 'mod',
                    'number'  : 'mod',
                    'prep'    : 'mod',
                    'poss'    : 'mod',
                    'possessive' : 'mod',
                    'prt'     : 'mod',
                'parataxis': 'dep',
                'punct'    : 'dep',
                'ref'      : 'dep',
                'sdep'     : 'dep',
                    'xsubj'   : 'sdep'
        }
