# -*- coding: UTF-8 -*-
import numpy as np


def parse_line(line):
    if line.startswith('#') or line == "\n":
        return None, None
    items = line.split(',')
    name = items[0]
    value = [float(items[i]) for i in range(1, len(items))]
    return name, value


# get the matrix X and Y from the file
def get_X_Y(data_file):
    f = open(data_file)
    lines = f.readlines()

    Y_name, Y = parse_line(lines[0])
    Y = np.array(Y)[:, np.newaxis]
    sample_num = Y.shape[0]
    
    X = [[1.]*sample_num, ]
    X_names = ["1", ]

    for i in range(1, len(lines)):
        name, value = parse_line(lines[i])
        if name is not None:
            assert(len(value) == sample_num)
            X.append(value)
            X_names.append(name)

    X = np.transpose(np.array(X))
    f.close()

    return Y_name, X_names, Y, X


# get the estimate-beta by the Least squares approximation
def get_est_beta(X, Y):
    XT = np.transpose(X)
    XTX = np.dot(XT, X)
    # if the inverse of XTX is not exist, give up
    assert(np.linalg.det(XTX) != 0)
    XTX_inv = np.linalg.inv(XTX)
    est_beta = np.dot(np.dot(XTX_inv, XT), Y)
    return est_beta, XTX_inv

# get the U and Q by the estimate-beta
def get_Q_U_F_R2(X, Y, est_beta):
    est_Y = np.dot(X, est_beta)
    print(est_Y)
    diff_Y = Y - est_Y
    print(diff_Y)
    diff_Y_T = np.transpose(diff_Y)
    Q = np.dot(diff_Y_T, diff_Y)[0,0]

    mean_Y = np.mean(Y)
    diff_mean_Y = est_Y - mean_Y
    diff_mean_Y_T = np.transpose(diff_mean_Y)
    U = np.dot(diff_mean_Y_T, diff_mean_Y)[0,0]

    n = Y.shape[0]
    p = X.shape[1] - 1
    F = (n - p - 1) / p * U / Q

    R2 = U / (Q + U)

    return Q, U, F, R2


# get t of beta i 
def get_t(beta_est, Q, X, Y, XTX_inv):
    n = Y.shape[0]
    p = X.shape[1] - 1
    cigma = np.sqrt(Q / (n - p - 1.))
    ts = [0.,]
    for i in range(1, beta_est.shape[0]):
        t = np.abs(beta_est[i, 0]  / ( cigma * np.sqrt(XTX_inv[i,i])))
        ts.append(t)
    return ts

def main():
    
    Y_name, X_names, Y, X = get_X_Y('data_GDP.csv')

    print("--------------\nData Format:\n--Samples(n):{} Features(p):{} ".format(Y.shape[0], X.shape[1] - 1))
    print("--Y_name:{}\n--X_names:{}\n--------------".format(Y_name, X_names))
    
    est_beta, XTX_inv = get_est_beta(X, Y)
    
    #print("Estimaet-Beta:\n{}".format(est_beta))

    Q, U, F, R2 = get_Q_U_F_R2(X, Y, est_beta)
    print("Q:{} U:{} F:{} R:{}".format(Q, U, F, R2))

    ts = get_t(est_beta, Q, X, Y, XTX_inv)
    
    for i in range(len(X_names)):
        print("beta:{:.5f}\tt:{:.3f}\t{}".format(est_beta[i,0], ts[i], X_names[i]))

if __name__ == "__main__":
    main()
