from Code.DataScience.fizz_buzz import binary_encode,fizz_buzz_encode
import numpy as np
import tqdm
from Code.ML.layers import Seqential,Linear,Loss,Tanh,Sigmoid,Layer,Momentum,SSE,SoftmaxCrossEntropy
xs = np.array([binary_encode(n) for n in range(101,1024)])
ys = np.array([fizz_buzz_encode(n) for n in range(101,1024)])

NUM_HIDDEN = 25

net = Seqential([Linear(input_dim=10,output_dim=NUM_HIDDEN,init="uniform"),
                 Tanh(),
                 Linear(input_dim=NUM_HIDDEN,output_dim=4,init="uniform"),
                 Sigmoid()])
net_softmax = Seqential([Linear(input_dim=10,output_dim=NUM_HIDDEN,init="uniform"),
                         Tanh(),
                         Linear(input_dim=NUM_HIDDEN,output_dim=4,init="uniform")])

def fizz_buzz_accuracy(low:int,hi:int,net:Layer):
    num_correct=0
    for n in range(low,hi):
        x = binary_encode(n)
        predicted = np.argmax(net.forward(x))
        actual = np.argmax(fizz_buzz_encode(n))
        if predicted == actual:
            num_correct+=1
    return num_correct/(hi-low)

"""  use sigmoid
optimizer = Momentum(learning_rate=0.1,momentum=0.9)
loss = SSE()

with tqdm.trange(1000) as t:
    for epoch in t:
        epoch_loss = 0.0
        for x,y in zip(xs,ys):
            predicted = net.forward(x)
            epoch_loss = loss.loss(predicted,y)
            gradient = loss.gradient(predicted,y)
            net.backward(gradient)
            optimizer.step(net)
        accuracy = fizz_buzz_accuracy(101,1024,net)
        t.set_description(f"fb loss : {epoch_loss:.2f} acc : {accuracy:.2f}")
print("test results", fizz_buzz_accuracy(1,101,net))
"""

# use softmax
optimizer = Momentum(learning_rate=0.1,momentum=0.9)
loss = SoftmaxCrossEntropy()
with tqdm.trange(1000) as t:
    for epoch in t:
        epoch_loss = 0.0
        for x,y in zip(xs,ys):
            predicted = net_softmax.forward(x)
            epoch_loss = loss.loss(predicted,y)
            gradient = loss.gradient(predicted,y)
            net_softmax.backward(gradient)
            optimizer.step(net_softmax)
        accuracy = fizz_buzz_accuracy(101,1024,net_softmax)
        t.set_description(f"fb loss : {epoch_loss:.2f} acc : {accuracy:.2f}")
print("test results", fizz_buzz_accuracy(1,101,net_softmax))

