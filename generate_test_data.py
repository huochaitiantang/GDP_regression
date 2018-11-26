import random

Y = []
X = [[], [], []]

cnt = 100

for i in range(cnt):
    x1 = random.random() * 10 - 5
    x2 = random.random() * 10 - 5
    x3 = random.random() * 10 - 5
    y = 5.42 + (3.14 * x1) + (-6.11 * x2) + (-0.97 * x3) + (random.random() - 0.5)
    #y = 5.42 + (3.14 * x1) + (random.random() - 0.5)
    X[0].append(x1)
    X[1].append(x2)
    X[2].append(x3)
    Y.append(y)

f = open("linear.csv", "w")
f.write("Y")
for i in range(cnt):
    f.write(",{:.3f}".format(Y[i]))
f.write("\nX1")
for i in range(cnt):
    f.write(",{:.3f}".format(X[0][i]))
f.write("\nX2")
for i in range(cnt):
    f.write(",{:.3f}".format(X[1][i]))
f.write("\nX3")
for i in range(cnt):
    f.write(",{:.3f}".format(X[2][i]))

f.close()
