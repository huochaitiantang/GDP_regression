# -*- coding: UTF-8 -*-
import sys
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

    def __init__(self, X, Y, X_names, Y_name, maps=None):
        self._X = X
        self._Y = Y
        self._n = Y.shape[0]
        self._p = X.shape[1] - 1
        self._X_names = X_names
        self._Y_name = Y_name
        if maps is None:
            self._maps = range(self._p + 1)
        else:
            self._maps = maps

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
            print("X{}\t| {:.5f}\t| {:.5f}\t| {:.5f}\t| {}".format(self._maps[i], self._est_beta[i,0], self._abs_Ts[i], self._Fs[i], self._X_names[self._maps[i]]))
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
        maps = []
        for i in range(len(self._X_names)):
            if i not in self._pop:
                maps.append(i)
        return fX,  maps

    def solve(self):
        step = 0
        while True:
            fX, maps = self.filter_X()
            model = RegressionModel(fX, self._Y, self._X_names, self._Y_name, maps)
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

        print("Backward End")
        model.visual(1978)



# Forward method for variable choice
class ForwardRegressionModel:
    
    def __init__(self, X, Y, X_names, Y_name, F_in_thresh):
        self._X = X
        self._Y = Y
        self._X_names = X_names
        self._Y_name = Y_name
        self._F_in = F_in_thresh
        self._p = X.shape[1] - 1
        # Variance inside and outside the model
        self._in = [0, ]
        self._out = [i for i in range(1, self._p + 1)]

    # filter the pop variance xi, get the reset feature
    def batch_filter_X(self):
        batch_fX = []
        batch_maps = []
        
        # Combine variance inside model and each variance outside the model
        for o in self._out:
            out = self._out.copy()
            out.remove(o)
            fX = np.delete(self._X, out, axis=1)

            maps = self._in.copy()
            maps.append(o)
            maps.sort()

            batch_fX.append(fX)
            batch_maps.append(maps)

        return batch_fX, batch_maps

    def solve(self):
        step = 0
        while True:
            batch_fX, batch_maps = self.batch_filter_X()
            max_F_new = max_o = 0
            for i in range(len(batch_fX)):
                fX = batch_fX[i]
                maps = batch_maps[i]

                model = RegressionModel(fX, self._Y, self._X_names, self._Y_name, maps)
                model.solve()
                model.format_result()
    
                # Get F value of new variance
                o = self._out[i]
                F_new = model.Fs()[maps.index(o)]
            
                if F_new > max_F_new:
                    max_F_new = F_new
                    max_o = o

            if max_F_new < self._F_in:
                break
                
            self._in.append(max_o)
            self._in.sort()
            self._out.remove(max_o)
            self._out.sort()

            step += 1
            print("\n* Step [ {} ]: Add X{}: {} Fi = {:.5f} >= {}".format(step, max_o, self._X_names[max_o], max_F_new, self._F_in))

        fX = np.delete(self._X, self._out, axis=1)
        model = RegressionModel(fX, self._Y, self._X_names, self._Y_name, self._in)
        model.solve()
        model.format_result()

        print("Fordward End")
        model.visual(1978)


# Forward and Backward method for variable choice
class ForwardBackwardRegressionModel:
    
    def __init__(self, X, Y, X_names, Y_name, F_in_thresh, F_out_thresh):
        self._X = X
        self._Y = Y
        self._X_names = X_names
        self._Y_name = Y_name
        self._F_in = F_in_thresh
        self._F_out = F_out_thresh
        self._p = X.shape[1] - 1
        # Variance inside and outside the model
        self._in = [0, ]
        self._out = [i for i in range(1, self._p + 1)]

    # filter the pop variance xi, get the reset feature
    def batch_filter_X(self):
        batch_fX = []
        batch_maps = []
        
        # Combine variance inside model and each variance outside the model
        for o in self._out:
            out = self._out.copy()
            out.remove(o)
            fX = np.delete(self._X, out, axis=1)

            maps = self._in.copy()
            maps.append(o)
            maps.sort()

            batch_fX.append(fX)
            batch_maps.append(maps)

        return batch_fX, batch_maps

    def solve(self):
        step = 0
        while True:
            batch_fX, batch_maps = self.batch_filter_X()
            max_F_new = max_o = 0
            max_Fs = max_maps = None
            for i in range(len(batch_fX)):
                fX = batch_fX[i]
                maps = batch_maps[i]

                model = RegressionModel(fX, self._Y, self._X_names, self._Y_name, maps)
                model.solve()
                model.format_result()
    
                # Get F value of new variance
                o = self._out[i]
                F_new = model.Fs()[maps.index(o)]
            
                if F_new > max_F_new:
                    max_F_new = F_new
                    max_o = o
                    max_Fs = model.Fs()
                    max_maps = maps

            if max_F_new < self._F_in:
                break
                
            self._in.append(max_o)
            self._in.sort()
            self._out.remove(max_o)
            self._out.sort()

            # Check if F value of original variance is too small
            for i in range(1, len(max_maps)):
                m = max_maps[i]
                if m != max_o and m in self._in:
                    if max_Fs[i] <= self._F_out:
                        self._in.remove(m)
                        self._in.sort()
                        self._out.append(m)
                        self._out.sort()
                        print("Remove     X{} {} F_value:{:.5f} <= F_out:{}".format(m, self._X_names[m], max_Fs[i], self._F_out))
                    else:
                        print("Not Remove X{} {} F_value:{:.5f} > F_out:{}".format(m, self._X_names[m], max_Fs[i], self._F_out))


            step += 1
            print("\n* Step [ {} ]: Add X{}: {} Fi = {:.5f} >= {}".format(step, max_o, self._X_names[max_o], max_F_new, self._F_in))

        fX = np.delete(self._X, self._out, axis=1)
        model = RegressionModel(fX, self._Y, self._X_names, self._Y_name, self._in)
        model.solve()
        model.format_result()

        print("Fordward-Backward End")
        model.visual(1978)


def main():
    #parseFile = ParseFile("linear.csv")
    parseFile = ParseFile("data_GDP.csv")

    # alpha = 0.10, F(1, 23) = 2.94
    # alpha = 0.05, F(1, 23) = 4.28
    # alpha = 0.01, F(1, 23) = 7.88
    F_in = float(sys.argv[2]) if len(sys.argv) > 2 else 2.94
    F_out = F_in
    assert(F_in > 0)

    regression_type = sys.argv[1] if len(sys.argv) > 1 else "full"
    assert(regression_type in ["full", "backward", "forward", "forward_backward"])

    if regression_type == "full":
        # Full model
        model = RegressionModel(parseFile.X(), parseFile.Y(), parseFile.X_names(), parseFile.Y_name())
        model.solve()
        model.format_result()
        model.visual(1978)

    elif regression_type == "backward":
        # Backward choose model
        model = BackwardRegressionModel(parseFile.X(), parseFile.Y(), parseFile.X_names(), parseFile.Y_name(), F_out)
        model.solve()

    elif regression_type == "forward":
        # Forward choose model
        model = ForwardRegressionModel(parseFile.X(), parseFile.Y(), parseFile.X_names(), parseFile.Y_name(), F_in)
        model.solve()

    else:
        # Forward choose model
        model = ForwardBackwardRegressionModel(parseFile.X(), parseFile.Y(), parseFile.X_names(), parseFile.Y_name(), F_in, F_out)
        model.solve()
    

if __name__ == "__main__":
    main()
