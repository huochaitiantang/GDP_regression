# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# Class for parsing file
class ParseFile:

    def __init__(self, file_path):
        f = open(file_path)
        self.parse_lines(f.readlines())
        f.close()

    def parse_line(self, line):
        if line.startswith('#') or line == "\n":
            return None, None
        items = line.split(',')
        value = [float(items[i]) for i in range(1, len(items))]
        return items[0], value

    # Y:[n x 1]
    # X:[n x (p + 1)]
    def parse_lines(self, lines):
        self._Y_name, Y = self.parse_line(lines[0])
        self._Y = np.array(Y)[:, np.newaxis]
        sample_num = self._Y.shape[0]

        X = [[1.]*sample_num, ]
        self._X_names = ["1", ]

        for i in range(1, len(lines)):
            name, value = self.parse_line(lines[i])
            if name is not None:
                assert(len(value) == sample_num)
                X.append(value)
                self._X_names.append(name)

        self._X = np.transpose(np.array(X))

    def Y(self):
        return self._Y

    def X(self):
        return self._X

    def Y_name(self):
        return self._Y_name

    def X_names(self):
        return self._X_names


# Class of linear regression model
class RegressionModel:

    def __init__(self, X, Y, X_names, Y_name):
        self._X = X
        self._Y = Y
        self._n = Y.shape[0]
        self._p = X.shape[1] - 1
        self._X_names = X_names
        self._Y_name = Y_name

        self._XTX_inv = None
        self._est_beta = None
        
        self._est_Y = None
        self._Q = None
        self._U = None
        self._R2 = None
        self._F = None

        self._abs_Ts = None
        self._Fs = None

    # Calculate the Estimate Beta by the Least Squares Approximation
    def cal_est_beta(self):
        XT = np.transpose(self._X)
        XTX = np.dot(XT, self._X)
        # If the inverse of XT*X is not exist, give up
        assert(np.linalg.det(XTX) != 0)
        self._XTX_inv = np.linalg.inv(XTX)
        # Est-Beta = (XT*X)^(-1) * XT * Y, shape:[(p + 1) x 1]
        self._est_beta = np.dot(np.dot(self._XTX_inv, XT), self._Y)

    # Calculate metric which reflects the effectiveness of model
    # est_Y =  X * est_beta
    # Q = (Y - est_Y)T * (Y - est_Y)
    # U = (est_Y - mean_Y)T * (est_Y - mean_Y)
    # R2 = U / L_yy = U / (U + Q)
    # F = (n - p - 1) / p * U / Q ~ F(p, n - p - 1)
    def cal_metric(self):
        self._est_Y = np.dot(self._X, self._est_beta)
        diff_Y = self._Y - self._est_Y
        diff_Y_T = np.transpose(diff_Y)
        self._Q = np.dot(diff_Y_T, diff_Y)[0, 0]

        mean_Y = np.mean(self._Y)
        diff_mean_Y = self._est_Y - mean_Y
        diff_mean_Y_T = np.transpose(diff_mean_Y)
        self._U = np.dot(diff_mean_Y_T, diff_mean_Y)[0,0]

        self._R2 = self._U / (self._Q + self._U)
        self._F = (self._n - self._p - 1) / self._p * self._U / self._Q

    # Check if beta[i] == 0
    # T[i] = est_beta[i] / (sqrt(Q / (n - p - 1)) * sqrt(XTX_inv[i][i]))
    # F[i] = est_beta[i]^2 / ((Q / (n - p - 1)) * XTX_inv[i][i])
    # T[i] ~ t(n - p - 1)
    # F[i] ~ F(1, n - p - 1)
    def cal_abs_Ts_Fs(self):
        cigma = np.sqrt(self._Q / (self._n - self._p - 1.))
        self._abs_Ts = [0., ]
        self._Fs = [0., ]
        for i in range(1, self._p + 1):
            t = np.abs(self._est_beta[i, 0] / (cigma * np.sqrt(self._XTX_inv[i,i])))
            self._abs_Ts.append(t)
            self._Fs.append(t * t)
    
    def solve(self):
        self.cal_est_beta()
        self.cal_metric()
        self.cal_abs_Ts_Fs()

    def X(self):
        return self._X

    def Y(self):
        return self._Y

    def n(self):
        return self._n

    def p(self):
        return self._p

    def est_beta(self):
        return self._est_beta

    def est_Y(self):
        return self._est_Y

    def Q(self):
        return self._Q

    def U(self):
        return self._U

    def R2(self):
        return self._R2

    def F(self):
        return self._F

    def abs_Ts(self):
        return self._abs_Ts

    def Fs(self):
        return self._Fs

    def format_result(self):
        print("n = {}, p = {}, n - p - 1 = {}".format(self._n, self._p, self._n - self._p - 1))
        print("Y: {}".format(self._Y_name))     
    
        print("-------------------------------Solve Result----------------------------------")
        print("Xi\t| EstBeta\t| T-Value\t| F-Value\t| X_name")
        print("-----------------------------------------------------------------------------")
    
        for i in range(self._p + 1):
            print("X{}\t| {:.5f}\t| {:.5f}\t| {:.5f}\t| {}".format(i, self._est_beta[i,0], self._abs_Ts[i], self._Fs[i], self._X_names[i]))
        print("-----------------------------------------------------------------------------")
    
        print("\tQ:  {:.5f}".format(self._Q))
        print("\tU:  {:.5f}".format(self._U))
        print("\tF:  {:.5f}".format(self._F))
        print("\tR2: {:.5f}".format(self._R2))
        print("\tmean-Q:  {:.5f}".format(self._Q / self._n))
        print("-----------------------------------------------------------------------------")

    def visual(self, start_number):
        # plot the estimate result
        x = np.arange(start_number, start_number + self._n)
        true_y = self._Y[:,0][::-1]
        est_y  = self._est_Y[:,0][::-1]
    
        plt.title('Estimated and True GDP')
        plt.xlabel('Year')
        plt.ylabel('GDP(100 million)')
    
        plt.plot(x, true_y, "r^-", label="true")
        plt.plot(x, est_y, "cv:", label="estimated")
        plt.legend(loc='upper left')
    
        plt.show()



# Backward method for variable choice
class BackwardRegressionModel:
    
    def __init__(self, X, Y, X_names, Y_name, F_out_thresh):
        self._X = X
        self._Y = Y
        self._X_names = X_names
        self._Y_name = Y_name
        self._F_out = F_out_thresh
        # Variance abandoned
        self._pop = []

    # filter the pop variance xi, get the reset feature
    def filter_X(self):
        fX = self._X.copy()
        if len(self._pop) > 0:
            fX = np.delete(fX, self._pop, axis=1)
        fX_names = []
        maps = []
        for i in range(len(self._X_names)):
            if i not in self._pop:
                fX_names.append(self._X_names[i])
                maps.append(i)
        return fX, fX_names, maps

    def solve(self):
        step = 0
        while True:
            fX, fX_names, maps = self.filter_X()
            #print(fX.shape, fX_names, self._pop)
            model = RegressionModel(fX, self._Y, fX_names, self._Y_name)
            model.solve()
            model.format_result()
            Fs = model.Fs()
            min_F_value = np.min(Fs[1:])
            min_F_index = np.argmin(Fs[1:]) + 1
            if min_F_value >= self._F_out:
                break
            origin_index = maps[min_F_index]
            self._pop.append(origin_index)
            self._pop.sort()
            step += 1
            print("\n* Step [ {} ]: Remove X{}: {} Fi = {:.5f} < {}".format(step, origin_index, self._X_names[origin_index], min_F_value, self._F_out))
        est_beta = model.est_beta()[:,0]
        abs_Ts = model.abs_Ts()
        Fs = model.Fs()
        print("---------------------------Backward Result-----------------------------------")
        print("Xi\t| Est-Beta\t| T-Value\t| F-Value\t| X_name")
        print("-----------------------------------------------------------------------------")
        for i in range(est_beta.shape[0]):
            print("X{}\t| {:.5f}\t| {:.5f}\t| {:.5f}\t| {}".format(maps[i], est_beta[i], abs_Ts[i], Fs[i], self._X_names[maps[i]]))
        print("-----------------------------Backward End------------------------------------")
        model.visual(1978)



def main():
    #parseFile = ParseFile("linear.csv")
    parseFile = ParseFile("data_GDP.csv")

    # All model
    '''
    model = RegressionModel(parseFile.X(), parseFile.Y(), parseFile.X_names(), parseFile.Y_name())
    model.solve()
    model.format_result()
    model.visual(1978)
    '''

    # Backward choose model
    # alpha = 0.05, F(1, 23) = 4.28
    # alpha = 0.01, F(1, 23) = 7.88
    F_out = 7.88
    model = BackwardRegressionModel(parseFile.X(), parseFile.Y(), parseFile.X_names(), parseFile.Y_name(), F_out)
    model.solve()


if __name__ == "__main__":
    main()
