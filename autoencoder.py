#/usr/bin/python2.7
#*-coding:utf-8-*-
import numpy as np
import theano
import theano.tensor as T
import theano.config.floatX as fX
class AutoEncoder(object):

	def __init__(self, numpy_rng, input=None, n_visible=400, n_hidden=200, W=None, bhid=None, bvis=None):
	'''
		numpy_rng: numpy.random.RandomState 
				used for weights generation
		input: theano.tensor.TensorType
				a symbolic description of the input or None 
		n_visible: int
				number of visible units
		n_hidden: int
				number of hidden units
		W: theano.tensor.TensorType
				a set of weights that should be shared 
		bhid: theano.tensor.TensorType
		bvis: theano.tensor.TensorType
				a set of biases values for hidden/visible units that should be shared
				
	'''
		self.n_visible = n_visible
		self.n_hidden  = n_hidden

		if not W:
			# W is initialized with 'initial_W' which is uniformely sampled
			# from a range in below.
			# theano.config.floatX so that the code is runable on GPU
			initial_W = np.asarray(numpy_rng.uniform(
						low   = -4*np.sqrt(6./(n_hidden + n_visible)),
						high  =  4*np.sqrt(6./(n_hidden + n_visible)),
						size  = (n_visible, n_hidden),
						dtype = fX
						)
			W = theano.shared(value=initial_W, name='W') #shared valiable.

		if not bvis:
			bvis = theano.shared(value=numpy.zeros(n_visible, dtype=fX))
		if not bhid:
			bhid = theano.shared(value=numpy.zeros(n_hidden,  dtype=fx))

		self.W      = W
		self.W_dash = self.W.T
		
		self.b      = bhid
		self.b_dash = bvis
		
		if not input:
			# if no input is given, generate a variable for input
			# use a matrix beacuse we expect a minibath of several examples
			# each example being a row (also for stochastic grad)
			
			self.x  = T.dmatrix(name='input')
		else:
			self.x  = input

		self.params = [self.W, self.b, self.b_dash]


	def get_hidden_values(self, input):
		''' computes the values of the hidden layer '''
		return T.tanh(T.dot(input, self.W) + self.b)
	

	def get_reconstructed(self, hidden):
		''' computes the values of the reconstructed layer '''
		return T.tanh(T.dot(hidden, self.W_dash) + self.b_dash)
	
	def get_cost(self):
		''' computes the standard reconstructed cost '''
		y = self.get_hidden_values(self.x)
		z = self.get_reconstructed(y)
		# If we are using minibatches, L will be a vector, with one entry per example in minibatch

		# cross-entropy cost should be modified here.
		L = -T.sum( (0.5*self.x+0.5)*T.log(0.5*z+0.5) + (-0.5*self.x+0.5)*T.log(-0.5*z+0.5) )
		# squred cost.
		#L = -T.sum( (self.x-z)**2 )
		
		cost = T.mean(L) + 0.01*(self.W**2).sum()   # cost for a minibatch
		return cost	



	def get_cost_updates(self, leaning_rate):
		''' computes the cost and the updates for one training step'''
		
		cost = self.get_cost()

		#computes the gradients of the cost respect to its parameters
		gparams = T.grad(cost, self.params)
		
		#generates the list of updates.
		updates = []
		for param, gpraram in zip(self.params, gparams):
			updates.append((param, param - learning_rate * gparam))

		return (cost, updates)
