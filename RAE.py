#/usr/bin/python2.7
#*-coding:utf-8-*-
import numpy as np
import theano
import theano.tensor as T
import copy 

class RAE(object):
    def __init__(self, numpy_rng, input=None, n_vector=200, W = None, bhid=None, bvis=None):
        self.n_vector  = n_vector
        self.n_visible = self.n_vector*2
        self.n_hidden  = self.n_vector
        
        if not W:
            # W is initialized with 'initial_W' which is uniformely sampled
            # from a range in below.
            # theano.config.floatX so that the code is runable on GPU
            initial_W = np.asarray(numpy_rng.uniform(
                        low   = -4*np.sqrt(6./(self.n_hidden + self.n_visible)),
                        high  =  4*np.sqrt(6./(self.n_hidden + self.n_visible)),
                        size  = (self.n_visible, self.n_hidden)),
                        dtype = theano.config.floatX)
            W = theano.shared(value=initial_W, name='W') #shared valiable.

        if not bvis:
            bvis = theano.shared(value=np.zeros(self.n_visible, dtype=theano.config.floatX))
        if not bhid:
            bhid = theano.shared(value=np.zeros(self.n_hidden,  dtype=theano.config.floatX))
        
        self.W      = W
        
        self.b      = bhid
        self.b_dash = bvis
        
        self.AEs = []
        for i in xrange(20):
            print 'AEs..',i
            self.AEs.append(AE(numpy_rng, input, n_vector, i+2 ,self.W, self.b, self.b_dash, rnn=True))
        



class AE(object):

    def __init__(self, numpy_rng, input=None, n_vector=200, n_num=2, W=None, bhid=None, bvis=None, rnn=False):
        '''
        numpy_rng: numpy.random.RandomState 
                used for weights generation
        input: an array of theano.tensor.TensorType
                a symbolic description of the input or None 
        n_vector: int
                length of word vector
        n_num: number of words input.
        W: theano.tensor.TensorType
                a set of weights that should be shared 
        bhid: theano.tensor.TensorType
        bvis: theano.tensor.TensorType
                a set of biases values for hidden/visible units that should be shared
        rnn: boolean
                a flag if using unfolding RAE
        '''
        self.rnn = rnn
        self.num = n_num
        
        self.n_vector  = n_vector

        self.n_visible = self.n_vector*2
        self.n_hidden  = self.n_vector

        if not W:
            # W is initialized with 'initial_W' which is uniformely sampled
            # from a range in below.
            # theano.config.floatX so that the code is runable on GPU
            initial_W = np.asarray(numpy_rng.uniform(
                        low   = -4*np.sqrt(6./(self.n_hidden + self.n_visible)),
                        high  =  4*np.sqrt(6./(self.n_hidden + self.n_visible)),
                        size  = (self.n_visible, self.n_hidden)),
                        dtype = theano.config.floatX)
            W = theano.shared(value=initial_W, name='W') #shared valiable.

        if not bvis:
            bvis = theano.shared(value=np.zeros(self.n_visible, dtype=theano.config.floatX))
        if not bhid:
            bhid = theano.shared(value=np.zeros(self.n_hidden,  dtype=theano.config.floatX))

        self.W      = W
        self.W_dash = self.W.T
        
        self.b      = bhid
        self.b_dash = bvis
        
        self.vector = None

        if not input:
            # if no input is given, generate a variable for input
            # use a matrix beacuse we expect a minibath of several examples
            # each example being a row (also for stochastic grad)
            
            self.x  = T.dvector(name='input')
        else:
            self.x  = input

        self.params = [self.W, self.b, self.b_dash]

        #Compile:
        self.compile()


    def reset_num(self, num):
        self.num = num

    def get_hidden_values(self, input):
        ''' computes the values of the hidden layer '''
        #return T.tanh(T.dot(T.reshape(input, (1, -1)), self.W) + self.b)
        return T.tanh(T.dot(input, self.W) + self.b)    

    def get_reconstructed(self, hidden):
        ''' computes the values of the reconstructed layer '''
        return T.tanh(T.dot(hidden, self.W_dash) + self.b_dash)
    
    def get_cost(self):  
        ''' computes the standard reconstructed cost (2 input)'''
        x = self.x
        y = self.get_hidden_values(x)
        
        # Save the hidden value as output vector
        self.vector = copy.deepcopy(y)
        
        z = self.get_reconstructed(y)
        # If we are using minibatches, L will be a vector, with one entry per example in minibatch

        # cross-entropy cost should be modified here.
        L = -T.sum( (0.5*x+0.5)*T.log(0.5*z+0.5) + (-0.5*x+0.5)*T.log(-0.5*z+0.5) )
        # squred cost.
        #L = -T.sum( (x-z)**2 )
        
        cost = T.mean(L) + 0.01*(self.W**2).sum()   # cost for a minibatch
        return cost 

    def get_unfolding_cost(self):
        ''' computes the unfolding rwconstructed cost (more than 2 inputs) '''
        x  = T.reshape(self.x, (-1, self.n_vector)) 
        yi = x[0];i=1
        for i in range(1, self.num):
        #while T.lt(i, self.num):
            xi = T.concatenate((yi, x[i]))
            yi = self.get_hidden_values(xi)
            i += 1
        # Save the deepest hidden value as output vactor
        self.vector = copy.deepcopy(yi)

        tmp = []
        i = 1
        for i in range(1, self.num):
        #while T.lt(i, self.num):
            zi = self.get_reconstructed(yi)
            t  = T.reshape(zi, (2, self.n_vector))
            tmp.append(t[1])
            yi = t[0]
            i += 1
        tmp.append(yi)
        tmp.reverse()
    
        x = self.x
        z = T.concatenate(tmp)
        
        # cross-entropy cost should be modified here.
        L = -T.sum( (0.5*x+0.5)*T.log(0.5*z+0.5) + (-0.5*x+0.5)*T.log(-0.5*z+0.5) )
        # squred cost.
        #L = -T.sum( (x-z)**2 )
        
        cost = T.mean(L) + 0.01*(self.W**2).sum()   # cost for a minibatch
        return cost 
        

    def get_cost_updates(self, learning_rate):
        ''' computes the cost and the updates for one training step'''
        if self.rnn == False:
            cost = self.get_cost()
        else:
            cost = self.get_unfolding_cost()

        #computes the gradients of the cost respect to its parameters
        gparams = T.grad(cost, self.params);    

        #generates the list of updates.
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam.astype(theano.config.floatX)))

        return (cost, updates)
    
    def get_vector(self):
        ''' return the vector of the auto-encoder. '''
        if not self.vector:
            return self.vector
        else:
            if self.rnn == False:
                x = self.x
                y = self.get_hidden_values(x)
                return y
            else:
                x  = T.reshape(self.x, (-1, self.n_vector))
                yi = x[0]
                i = 1
                for i in range(1, self.num):
                #while T.lt(i, self.num):
                    xi = T.concatenate((yi, x[i]))
                    yi = self.get_hidden_values(xi)
                    i += 1
                return yi
            pass
        pass
    
    def compile(self):
        print 'compile...'
        x = T.dvector('x')
        
        self.x = x
        cost, updates = self.get_cost_updates(learning_rate=0.01)
        vector = self.get_vector()
        
        self.train    = theano.function([x], [cost], updates=updates)
        self.predict  = theano.function([x], [vector])
        self.countcost= theano.function([x], [cost])



