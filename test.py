#!/usr/bin/env python
# coding=utf-8

import theano
from theano import tensor as T
from multiprocessing import Pool
import os
import time 
import datetime
import cPickle as pickle
def f(x,fs):
    print os.getpid(), datetime.datetime.now()
    k = 0
    for i in xrange(100000):
        fs[x](x)
    print os.getpid(), datetime.datetime.now()
    return x

def f2(x):
    print datetime.datetime.now()
    time.sleep(1)
    for i in xrange(1000000):
        pass
    return x

def compile():
    print 'compile'
    x  = T.iscalars('x')
    z  = x**10
    fi = theano.function([x],z)
    return fi

if __name__== "__main__":
    fs = []
    
    pool = Pool(24)
    print 'start'
    #fs =  [compile() for xi in xrange(6)]
    file = open('../data/test.db', 'rb')
    fs = pickle.load(file)
    file.close()
    result = [x.get() for x in [pool.apply_async(f, (xi, fs)) for xi in xrange(6)]]
    print result
