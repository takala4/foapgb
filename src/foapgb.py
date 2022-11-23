"""Module summary.

This is core.py.
"""


import numpy as np
import scipy.optimize as optimize
import scipy.sparse as spsp

from collections import defaultdict

class Parameter:
    '''モデルのパラメータを保持するクラス

    Attributes
    ----------
    K: int
        立地点数
    tau: float
        企業間交流費用パラメータ
    t: float 
        通勤費用パラメータ
    L: float 
        労働需要パラメータ
    S_total: float 
        都市の総床面積
    theta_firm: float
        企業のランダム項パラメータ 
    theta_house: 
        家計のランダム項パラメータ
    T: numpy.ndarray (K, K)
        通勤費用行列
    Ker: numpy.ndarray (K, K)
        Tのカーネル行列
    D: numpy.ndarray (K, K)
        空間割引行列
    S: numpy.ndarray (K, )
        各立地点の床面積
    M: float
        総企業数
    N: float
        総家計数
    '''
    
    def __init__(self, K, distance_matrix, t, tau, L, S_total, theta_firm, theta_house):
        '''
        Parameters
        ----------
        K: int
            立地点数
        distance_matrix: numpy.ndarray (K, K)
            距離行列
        tau: float
            企業間交流費用パラメータ
        t: float 
            通勤費用パラメータ
        L: float 
            労働需要パラメータ
        S_total: float 
            都市の総床面積
        theta_firm: float
            企業のランダム項パラメータ 
        theta_house: 
            家計のランダム項パラメータ
        '''
      
        self.K = K
        self.tau = tau
        self.t = t 
        self.L = L
        self.S_total = S_total
        self.theta_firm = theta_firm
        self.theta_house = theta_house

        self.T = t*distance_matrix
        self.Ker = np.exp(-self.theta_house*self.T)
        self.D = np.exp(-tau*distance_matrix)
        self.S = np.array([S_total/K for i in range(K)])
        self.M = S_total*(1/(1+L))
        self.N = S_total*(L/(1+L))
    
class Sub:
    '''サブ問題クラス
    Attributes
    ----------
    prm: Parameter
       パラメータクラス 
    '''

    def __init__(self, prm):
        '''
        Parameters
        ----------
        prm: Parameter
            パラメータクラス
        '''
        self.prm = prm
    
    
    def set_m(self, m):
        '''企業分布の初期値を設定する関数
        Parameters
        ----------
        m: numpy.ndarray (K, )
            企業分布
        '''
        self.m = m
    
    def isFeasible(self):
        '''サブ問題の制約条件を満たしているかを判定する関数
        Returns
        -------
        flag: bool
            制約条件を満たしているかどうか
        '''

        flag = True
        # Demand side conservation        
        if not (self.m.sum()-self.prm.M < 10**(-6)):
            print('Demand not satisfied')
            flag = False
        
        # Supply side conservation
        if not (np.all(self.m <= self.prm.S_total/self.prm.K)):
            print('Supply not satisfied')
            print(self.prm.S_total/self.prm.K-self.m)
            flag = False

        return flag
    
    def solve(self, max_itr=10, err=10**(-6)):
        '''サブ問題を解く関数
        Parameters
        ----------
        max_itr: int
            最大反復回数
        err: float
            収束判定の閾値
        Returns
        -------
        u: numpy.ndarray (K, )
            u
        v: numpy.ndarray (K, )
            v
        gamma: float
            u@self.prm.Ker@v
        '''

        if self.isFeasible()==False:
            raise ValueError('Sub-problem is not feasible')
        
        self.gamma_hist = defaultdict(int)
        self.gamma_hist[-1] = 0.0

        u = np.ones(self.prm.K)
        v = np.ones(self.prm.K)

        
        O = (self.prm.L*self.m)/self.prm.N
        D = (self.prm.S-self.m)/self.prm.N

        for i in range(max_itr):
            self.gamma_hist[i] = u@self.prm.Ker@v

            v = self.gamma_hist[i] * (O/(self.prm.Ker@u))
            u = self.gamma_hist[i] * (D/(self.prm.Ker@v))

            if abs(self.gamma_hist[i]-self.gamma_hist[i-1]) < err:
                break                        
      
        gamma = self.gamma_hist[i]

        if np.all(u > 0) == False:
            print('v=', v)
            print('u=', u)
            raise ValueError('u includes non-positive values')

        if np.all(v > 0) == False:
            print('v=', v)
            print('u=', u)
            raise ValueError('v includes non-positive values')
            
        self.u = u
        self.v = v
        self.gamma = gamma
        return u, v, gamma
    
    def n(self, u, v, gamma):
        '''家計の通勤パターンを計算する関数
        Parameters
        ----------
        u: numpy.ndarray (K, )
            u
        v: numpy.ndarray (K, )
            v
        gamma: float
            gamma
        Returns
        -------
        n: numpy.ndarray (K, )
            家計の通勤パターン
        '''
        return (self.prm.N/gamma)*((u*self.prm.Ker).T*v).flatten()
    
    def primal_f(self, u, v, gamma):
        '''サブ問題の目的関数を計算する関数
        Parameters
                ----------
        u: numpy.ndarray (K, )
            u
        v: numpy.ndarray (K, )
            v
        gamma: float
            gamma
        Returns
        -------
        Z_primal : float
            サブ問題の目的関数値
        '''
        n = self.n(u, v, gamma)
        Z_primal = self.prm.T.flatten()@n + (1/self.prm.theta_house)*(n@np.log(n/self.prm.N))
        
        return Z_primal

    def dual_f(self, u, v, gamma):
        '''サブ問題の双対問題の目的関数を計算する関数
        Parameters
        ----------
        u: numpy.ndarray (K, )
            u
        v: numpy.ndarray (K, )
            v
        gamma: float
            gamma
        Returns
        -------
        Z_dual : float
            サブ問題の双対問題の目的関数値
        '''

        W = (1/self.prm.theta_house)*np.log(v)
        R = -(1/self.prm.theta_house)*np.log(u)
        Z_dual = - (self.prm.S-self.m)@R + (self.prm.L*self.m)@W - (self.prm.N/self.prm.theta_house)*np.log(gamma)
        
        return Z_dual
    
    
class Master:
    '''マスター問題を解くクラス
    Attributes
    ----------
    prm: Parameters
        パラメータ
    sub: Sub
        サブ問題クラス
    '''

    def __init__(self, prm, sub):
        '''
        Parameters
        ----------
        prm: Parameters
            パラメータ
        sub: Sub
            サブ問題クラス
        '''
        self.prm = prm
        self.sub = sub
        self.sub_itr_max = 20
    
    def calc_WR(self, u, v, gamma):
        '''W, Rを計算する関数
        Parameters
        ----------
        u: numpy.ndarray (K, )
            u
        v: numpy.ndarray (K, )
            v
        gamma: float
            gamma
        '''
        
        W = (1/self.prm.theta_house)*np.log(v)
        R = -(1/self.prm.theta_house)*np.log(u)
                        
        return W, R
            
    def n(self,u,v,gamma):
        '''家計の通勤パターンを計算する関数
        Parameters
        ----------
        u: numpy.ndarray (K, )
            u
        v: numpy.ndarray (K, )
            v
        gamma: float
            gamma
        Returns
        -------
        n: numpy.ndarray (K, )
            家計
        '''

        n = (self.prm.N/gamma)*((u*self.prm.Ker).T*v).flatten()
        return n
    
    def F(self, m):
        '''目的関数を計算する関数
        Parameters
        ----------
        m: numpy.ndarray (K, )
            企業分布
        Returns
        -------
        F: float
            目的関数値
        '''
        self.sub.set_m(m)
        u, v, gamma = self.sub.solve(max_itr=self.sub_itr_max)
        Sub = self.sub.dual_f(u, v, gamma)
        
        F_value = -(1/2)*m@self.prm.D@m \
        + (1/self.prm.theta_firm)*(m@np.log(m/self.prm.M))\
        + Sub
        
        return F_value

    
    def dFm(self, m):
        '''目的関数の勾配を計算する関数
        Parameters
        ----------
        m: numpy.ndarray (K, )
          企業分布
        Returns
        -------
        dFm: numpy.ndarray (K, )
            目的関数の勾配
        '''
        self.sub.set_m(m)
        u, v, gamma = self.sub.solve(max_itr=self.sub_itr_max)
        W, R= self.calc_WR(u, v, gamma)
        
        dFm_value = - self.prm.D@m \
        + (1/self.prm.theta_firm)*(np.log(m/self.prm.M) + np.ones(self.prm.K))\
        + self.prm.L*W + R
        
        return dFm_value
        
    

    def F_and_dFm(self, m):
        '''目的関数と勾配を計算する関数
        Parameters
        ----------
        m: numpy.ndarray (K, )
            企業分布
        Returns
        -------
        F: float
            目的関数値
        dFm: numpy.ndarray (K, )
            目的関数の勾配
        '''

        self.sub.set_m(m)
        u, v, gamma = self.sub.solve(max_itr=self.sub_itr_max)

        Sub = self.sub.dual_f(u, v, gamma)
        
        F_value = -(1/2)*m@self.prm.D@m \
        + (1/self.prm.theta_firm)*(m@np.log(m/self.prm.M))\
        + Sub

        W, R= self.calc_WR(u, v, gamma)
        
        dFm_value = - self.prm.D@m \
        + (1/self.prm.theta_firm)*(np.log(m/self.prm.M) + np.ones(self.prm.K))\
        + self.prm.L*W + R

        return F_value, dFm_value

    def solve(self, m0=None, err=10**(-3), max_itr=100, Lip=10):
        '''マスター問題を解く関数
        Parameters
        ----------
        m0: numpy.ndarray (K, )
            初期値
        err: float
            許容誤差
        max_itr: int
            最大反復回数
        Lip: float
            初期リプシッツ定数
        Returns
        -------
        m_hist: dict {i:numpy.ndarray (K, )}
            {反復回数: 企業分布}
        F_hist: dict {i:float}
            {反復回数: 目的関数値}
        '''

        s_min=10**(-5)
        s_max=self.prm.S[0]-10**(-5)
        S=self.prm.M
        

        alpha_hist = {}
        beta_hist  = {}
        lambda_hist = {}

        m_md_hist = {}
        m_ag_hist = {}
        m_hist = {}
        
        m_ag_hist[0] = m0
        m_md_hist[0] = m0
        m_hist[0] = m0

        F_hist = {}
        F_hist[0] = self.F(m0)

        dFm_md_hist = {}
        dFm_md_hist[0] = self.dFm(m0)

        for k in range(1, max_itr):
            
 
            alpha_hist[k] = 2/(k+1)
            
            m_md_hist[k] = (1 - alpha_hist[k])*m_ag_hist[k-1] + alpha_hist[k]*m_hist[k-1]
            dFm_md_hist[k] = self.dFm(m_md_hist[k])


            if k==1 :
                L = Lip
            else:
                L = max(1, np.linalg.norm(dFm_md_hist[k] - dFm_md_hist[k-1])/np.linalg.norm(m_md_hist[k] - m_md_hist[k-1]))
            
            beta_hist[k] = 1/(2*L)
            lambda_hist[k] = (1 + alpha_hist[k]/4)*beta_hist[k]
            
            m_hist[k] = Projection_CappedSimplex((m_hist[k-1] - lambda_hist[k]*dFm_md_hist[k]), s_min=s_min, s_max=s_max, S=S)

            m_ag_hist[k] = Projection_CappedSimplex((m_md_hist[k]-beta_hist[k]*dFm_md_hist[k]), s_min=s_min, s_max=s_max, S=S)
                          
            if (k!=1) and (np.linalg.norm(m_md_hist[k]-m_ag_hist[k])<err):
                break
        
        return m_ag_hist, F_hist 
    
    
    def isFeasible(self,m):
        '''マスター問題の解が制約条件を満たすかどうかを判定する関数
        Parameters
        ----------
        m: numpy.ndarray (K, )
            企業分布
        Returns
        -------
        isFeasible: bool
            制約条件を満たすかどうか
        '''
        return bool(np.all(m<=self.prm.S_total/self.prm.K)*(abs(m.sum()-self.prm.M)<10**(-6)))

    def make_random_m(self):
        '''ランダムな企業分布を生成する関数
        Returns
        -------
        m: numpy.ndarray (K, )
            企業分布
        '''
        r = np.random.rand(self.prm.K)*(self.prm.S_total/self.prm.K)
        return self.prm.M*(r/(r.sum()))


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

  

def Projection_CappedSimplex(y, s_min=0, s_max=1, S=1):
    """Projection onto the capped simplex
    
    min_x \|x - y\|_2^2 s.t. x \in [s_min, s_max]^n, <1,x> = S 
    input: 
    y - array of length n
    s_min - lower bound ( 0 <= s_min )
    s_max - upper bound (s_min < s_max)
    S - total constraint parameter  \in [s_min*n, s_max*n]

    Reference:
    Weiran Wang: "Projection onto the capped simplex". March 3, 2015, arXiv:1503.01002.

    Parameters
    ----------
    y : array
        射影前の点
    s_min : float
        最小値
    s_max : float
        最大値
    S : float
        合計    
            
    Returns
    -------
    x : array
        射影後の点
    """
            
    # Scaling the problem to the following form
    # min_x \|x' - y' \|_2^2 s.t. x \in [0, 1]^n, <1,x> = S' 
    
    size = y.shape[0]
    
    # Scale: Mimimum -> 0
    Scaled_S = S - s_min*size
    Scaled_s_max = s_max - s_min
    y = y - s_min
    
    # Scale: Maximum -> 1
    scaling_parameter = 1/Scaled_s_max
    Scaled_S = Scaled_S*scaling_parameter
    y=y*scaling_parameter
    
    # reScale function for return
    reScale = lambda x: x/scaling_parameter + s_min
    
    
    # Start the reffered paper's algorithm    
    # Check some base case
    if Scaled_S > size :
        raise ValueError('problem is not feasible')
    elif Scaled_S==0:
        return reScale(np.zeros(size))
    elif Scaled_S==size:
        return reScale(np.ones(size))
        
    # Sort and concatenate to get -infty, y_1, y_2, ..., y_{n}, +infty
    idx = np.argsort(y)
    
    ys = np.concatenate(([-np.inf],y[idx],[np.inf]))
    x = np.zeros(size)
    
    # cumsum and concatenate
    T = np.concatenate(([0],np.cumsum(ys[1:])))

    # main loop a = 0, ..., n+1
    for a in range(0, size+2):
        if Scaled_S == (size - a) and ys[a+1] - ys[a] >= 1:
            b = a
            x[idx] = np.concatenate((np.zeros(a),ys[a+1:b+1] + gamma,np.ones(size-b)))
            return reScale(x)
        # inner loop b = a+1, ..., n
        for b in range(a+1,size+1):
            gamma = (Scaled_S + b - size + T[a] - T[b])/(b - a)
            if (ys[a] + gamma <= 0) and (ys[a+1] + gamma > 0) and (ys[b] + gamma < 1) and (ys[b+1] + gamma >= 1):
                x[idx] = np.concatenate((np.zeros(a),ys[a+1:b+1] + gamma,np.ones(size-b)))
                return reScale(x)