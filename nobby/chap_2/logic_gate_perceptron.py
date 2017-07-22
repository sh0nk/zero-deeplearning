import numpy as np
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x*w) + b
    return 0 if tmp <= 0 else 1
    
def AND(x1, x2):
    s1 = NAND(x1, x2)
    return NAND(s1, s1)

def OR(x1, x2):
    s1 = NAND(x1, x1)
    s2 = NAND(x2, x2)
    return NAND(s1, s2)

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))