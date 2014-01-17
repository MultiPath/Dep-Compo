import numpy as np
import theano
import theano.tensor as T

N     = 400
feats = 784
rng = np.random
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_step = 1000

#Declare Theano symbolic variables
x = T.matrix('x')
y = T.vector('y')
w = theano.shared(rng.randn(feats), name='w')
b = theano.shared(0.0, name='b')
print "initial model"
#print len(w.get_value())
#print b.get_value()

#Construct Theano expression graph
p_1 = 1/(1 + T.exp(-T.dot(x, w) - b))    #probability that target = 1
prediction = p_1 > 0.5                   #The prediction thresholded
xent = -y*T.log(p_1) - (1 - y)*T.log(1 - p_1)  #Cross-entropy loss function
cost = xent.mean() + 0.01*(w**2).sum()   #The cost to minimize (lambda=0.01, L2)
gw, gb = T.grad(cost, [w, b])            #Compute the gradient of the cost
print 'construct ok.'

#Compile
train = theano.function(
		inputs  = [x, y],
		outputs = [prediction, xent],
		updates = ((w, w - 0.1*gw), (b, b - 0.1*gb)))  #0.1 is the learning step
predict = theano.function(inputs = [x], outputs = prediction)
print 'compile ok.'


#Train
for i in range(training_step):
	if i%100 == 0:
		print 'round',i
	pred, err = train(D[0], D[1])
print 'train ok.'


print "final model"
#print w.get_value(), b.get_value()
print "target values for D:", D[1]
print "prediction on D:", predict(D[0])

