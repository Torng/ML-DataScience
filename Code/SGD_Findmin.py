import random

def fx(x:float)->float:
    return x**4-50*(x**3)-x-1
def dfx(x:float)->float:
    return 4*(x**3)-150*(x**2)-1

guess_x = random.randint(0,10)
# guess_x = 24
learning_rate = 0.00004
exgrad = 0
for epoch in range(10000):
    grad = dfx(guess_x)
    guess_x = guess_x-grad*learning_rate
    print(epoch,guess_x)
    # print(round(theta[0],3),round(theta[1],3))
