"""Module summary.

This is test.py.
"""


import numpy as np
import scipy.optimize as optimize
import scipy.sparse as spsp

class Test:
    
    def __init__(self, prm, m, u, v, gamma):
        self.prm = prm
        self.m = m
        self.u = u
        self.v = v
        self.gamma = gamma

        self.pre_W = (1/self.prm.theta_house)*np.log(v)
        self.pre_R = -(1/self.prm.theta_house)*np.log(u)

        self.W = self.pre_W - self.pre_W.min()
        self.R = self.pre_R - self.pre_R.min()


        R_matrix = np.dot(
            np.ones(shape=(self.prm.K, self.prm.K)), np.diag(self.R)).T
        W_matrix = np.dot(
            np.ones(shape=(self.prm.K, self.prm.K)), np.diag(self.W))

        self.expC_n = np.exp(-self.prm.theta_house*(- W_matrix + \
                                        self.prm.T + R_matrix)).flatten()

        self.Sum_expC_n = self.expC_n.sum()

        self.ExMinCost_n = (self.prm.N/self.prm.theta_house)*(np.log(self.Sum_expC_n))

        self.expC_m = np.exp(-self.prm.theta_firm*(- self.prm.D@self.m + \
                                        self.R + self.prm.L*self.W))
        self.Sum_expC_m = self.expC_m.sum()
        self.ExMinCost_m = (self.prm.M/self.prm.theta_firm)*(np.log(self.Sum_expC_m))

        self.n = (self.prm.N/gamma)*((u*self.prm.Ker).T*v).flatten()
        self.n_matrix = (self.prm.N/gamma)*((u*prm.Ker).T*v)

    def primal(self):
        Primal = -(1/2)*(self.m@self.prm.D@self.m) \
            + self.prm.T.flatten()@self.n \
            + (1/self.prm.theta_firm)*(self.m@np.log(self.m/self.prm.M))\
            + (1/self.prm.theta_house)*(self.n@np.log(self.n/self.prm.N))

        return Primal

    def dual(self):
        return - self.ExMinCost_m - self.ExMinCost_n + (1/2)*(self.m@self.prm.D@self.m) - self.prm.S@self.R

    def check_n(self, err=10**(-6)):

        return np.linalg.norm(self.n - self.prm.N*(self.expC_n/(self.Sum_expC_n)))**2

    def check_m(self, err=10**(-6)):

        return np.linalg.norm(self.m - self.prm.M*(self.expC_m/(self.Sum_expC_m)))**2 

    def check_m_cnsv(self, err=10**(-6)):
        return np.linalg.norm(self.m.sum() - self.prm.M)**2 

    def check_n_cnsv(self, err=10**(-6)):
        return np.linalg.norm(self.n.sum() - self.prm.N)**2 

    def check_Land(self, err=10**(-6)):
        n_matrix = self.n.reshape(self.prm.K, self.prm.K)
        n_population = n_matrix@np.ones(self.prm.K)
        return np.linalg.norm(self.prm.S - n_population - self.m)**2 

    def check_Labor(self, err=10**(-6)):

        n_Labor = self.n_matrix.T@np.ones(self.prm.K)
        return np.linalg.norm(n_Labor-self.prm.L*self.m)**2 

    def check_all(self, err=10**(-6)):
        print('check_n',      self.check_n(err) < err)
        print('check_m',      self.check_m(err) < err)
        print('check_m_cnsv', self.check_m_cnsv(err) < err)
        print('check_n_cnsv', self.check_n_cnsv(err) < err)
        print('check_Land',   self.check_Land(err) < err)
        print('check_Labor',  self.check_Labor(err) < err)
        return print('Cheaked')

    def ChoiceProb_m(self):
        expCm = np.exp(-self.prm.theta_firm*(- self.prm.D@
                                        self.m + self.R + self.prm.L*self.W))
        return self.prm.M*(expCm/(expCm@np.ones(self.prm.K)))