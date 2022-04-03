from Code.ML.layers import Seqential,Linear,Sigmoid,GradientDescent,SSE,SoftmaxCrossEntropy,Tanh,Momentum
import tqdm
net = Seqential([Linear(input_dim=2,output_dim=10),
                 Tanh(),
                 Linear(input_dim=10,output_dim=1)])

xs = [[0.,0.],[0.,1.],[1.,0.],[1.,1.]]
ys = [[0.],[1.],[1.],[0.]]

optimizer = Momentum(learning_rate=0.1)
loss = SoftmaxCrossEntropy()

with tqdm.trange(3000) as t:
    for epoch in t :
        epoch_loss = 0.0

        for x,y in zip(xs,ys):
            predicted = net.forward(x)
            epoch_loss += loss.loss(predicted,y)
            gradient = loss.gradient(predicted,y)
            net.backward(gradient)
            optimizer.step(net)

        t.set_description(f"xor loss{epoch_loss:.3f}")


print(net.forward([0,0]))
print(net.forward([0,1]))
print(net.forward([1,0]))



