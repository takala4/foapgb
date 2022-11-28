'''
描画モジュール
'''

import numpy as np
import matplotlib.pyplot as plt

    
def m_image(prm, m, max_flag=True):
    '''企業分布を描画する関数

    Parameters
    ----------
    prm : Parameter
        パラメータクラス
    m : numpy.ndarray
        企業分布
    max_flag : bool, optional
        |配色の最大値を床面積の最大値にするかどうか
        |True  = 床面積の最大値が最も濃い赤
        |False = mの最大値が最も濃い赤
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

    