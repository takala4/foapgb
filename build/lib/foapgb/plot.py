"""Module summary.

This is plot.py.
"""


import numpy as np
import scipy.optimize as optimize
import scipy.sparse as spsp
import matplotlib.pyplot as plt

    
def m_image(prm, m, max_flag=True):
    '''企業分布を描画する関数

    Parameters
    ----------
    prm : class
        パラメータクラス
    m : array
        企業分布
    max_flag : bool, optional
        配色の最大値を床面積の最大値にするかどうか, by default True
    '''
    Num_Cols = int(np.sqrt(prm.K))
    mat = np.reshape(m, (Num_Cols, Num_Cols))
    
    plt.figure(figsize=(5,5))
    if max_flag:
        plt.imshow(mat, interpolation='nearest', vmin=0.0, vmax=prm.S.max(), cmap='bwr')
    else:
        plt.imshow(mat, interpolation='nearest', cmap='bwr')
        
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    
    # plt.colorbar()