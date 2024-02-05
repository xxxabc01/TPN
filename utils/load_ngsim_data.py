import numpy as np
import pickle
import scipy.io
import pandas as pd


class ngsim_data_loader():

    def __init__(self, Loop_number):

        self.N_loop = Loop_number

    def load_test(self):
        return self.X_star, self.Exact_rho, self.Exact_u
    def load_data(self):
        frame = scipy.io.loadmat('data/data.mat')  # use as frame of x and t

        # example of loop dict based on data
        Loop_dict = {8:[0, 11, 22, 33, 45, 56, 67, 79]
                     }

        t = frame['tScale'].T.flatten()[:, None]
        x = frame['xScale'].T.flatten()[:, None]
        vel = pd.read_table('data/NGSIM_US80_4pm_Velocity_Data.txt', delim_whitespace=True)
        den = pd.read_table('data/NGSIM_US80_4pm_Density_Data.txt', delim_whitespace=True)

        # reshape the data
        x = (x[:vel.shape[0]] / 5 * 20).astype(int)  # 20 span
        t = (t[:vel.shape[1]] * 5).astype(int)  # 5 span
        Exact = np.real(vel.T)

        """
        rho data
        """

        Exact_rho = np.real(den.T)


        X, T = np.meshgrid(x, t)
        # print(len(X), len(X[0]))  # shape of the data

        N_f = int(len(X) * len(X[0]) * 0.8)


        x = X.flatten()[:, None]
        t = T.flatten()[:, None]


        self.Exact_rho = Exact_rho
        self.Exact_u = Exact
        N_loop = Loop_dict[self.N_loop]

        X_star = np.hstack((x, t))
        self.X_star = X_star.astype(np.float32)

        idx = np.random.choice(X_star.shape[0], N_f, replace=False)
        idx2 = []

        for i in range(Exact_rho.shape[0]):
            base = i * Exact_rho.shape[1]
            index = [base + ele for ele in N_loop]
            idx2 += index

        self.rho_star = Exact_rho.flatten()[:, None]
        self.u_star = Exact.flatten()[:, None]

        # low and up bound
        self.lb = X_star.min(0)
        self.ub = X_star.max(0)

        self.rho_train = self.rho_star[idx2, :]

        self.u_train = self.u_star[idx2, :]

        self.X_u_train =self.X_star[idx2,:]
        self.X_f_train = self.X_star[idx,:]
