'''
テストモジュール
'''

import numpy as np
import scipy.sparse as spsp

class Test:
    '''テストクラス
    
    Attributes
    ----------
    prm : Parameter
        パラメータクラス
    m : numpy.ndarray
        企業分布
    u : numpy.ndarray
        u
    v : numpy.ndarray
        v
    gamma : float
        gamma
    '''

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
        '''主問題の目的関数を計算する関数
        
        Returns
        -------
        primal_value: float
            主問題の目的関数値
        '''
        primal_value = -(1/2)*(self.m@self.prm.D@self.m) \
            + self.prm.T.flatten()@self.n \
            + (1/self.prm.theta_firm)*(self.m@np.log(self.m/self.prm.M))\
            + (1/self.prm.theta_house)*(self.n@np.log(self.n/self.prm.N))

        return primal_value

    def dual(self):
        '''双対問題の目的関数を計算する関数

        Returns
        -------
        dual_value: float
            双対問題の目的関数値
        '''
        dual_value = - self.ExMinCost_m - self.ExMinCost_n + (1/2)*(self.m@self.prm.D@self.m) - self.prm.S@self.R
        return dual_value

    def check_n_prob(self):
        '''家計の選択確率式のチェック関数

        Returns
        -------
        check_value: float
            家計の選択確率式のチェック値
        '''

        check_value = np.linalg.norm(self.n - self.prm.N*(self.expC_n/(self.Sum_expC_n)))**2

        return check_value

    def check_m_prob(self):
        '''企業の選択確率式のチェック関数

        Returns
        -------
        check_value: float
            企業の選択確率式のチェック値
        '''

        check_value = np.linalg.norm(self.m - self.prm.M*(self.expC_m/(self.Sum_expC_m)))**2

        return check_value

    def check_m_cnsv(self):
        '''企業の保存条件のチェック関数

        Returns
        -------
        check_value: float
            企業の保存条件のチェック値
        '''

        check_value = np.linalg.norm(self.m.sum() - self.prm.M)**2
        return check_value

    def check_n_cnsv(self):
        '''家計の保存条件のチェック関数

        Returns
        -------
        check_value float
            家計の保存条件のチェック値
        '''
        
        check_value =  np.linalg.norm(self.n.sum() - self.prm.N)**2 
        return check_value

    def check_Land(self):
        '''土地市場の清算条件のチェック関数
        
        Returns
        -------
        check_value: float
            土地市場の清算条件のチェック値
        '''
        n_matrix = self.n.reshape(self.prm.K, self.prm.K)
        n_population = n_matrix@np.ones(self.prm.K)
        check_value = np.linalg.norm(self.prm.S - n_population - self.m)**2 
        return check_value

    def check_Labor(self):
        '''労働市場の清算条件のチェック関数

        Returns
        -------
        check_value: float
            労働市場の清算条件のチェック値
        '''
        n_Labor = self.n_matrix.T@np.ones(self.prm.K)
        check_value = np.linalg.norm(n_Labor-self.prm.L*self.m)**2 
        return check_value

    def check_all(self, err=10**(-6)):
        '''全ての均衡条件をチェックする関数

        各条件が許容誤差以内であればTrueを，そうでなければFalseをprintする

        Parameters
        ----------
        err: float
            許容誤差
        '''
        print('check_n_prob',      self.check_n_prob() < err)
        print('check_m_prob',      self.check_m_prob() < err)
        print('check_m_cnsv',      self.check_m_cnsv() < err)
        print('check_n_cnsv',      self.check_n_cnsv() < err)
        print('check_Land'  ,      self.check_Land()   < err)
        print('check_Labor' ,      self.check_Labor()  < err)
