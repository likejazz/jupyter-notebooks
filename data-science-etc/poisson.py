# %%
import numpy as np
from scipy.special import factorial
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns


def poisson(l):
    """
    푸아송 분포(poisson distribution)
    """
    global r

    return (np.e ** -l * l ** r) / factorial(r)


r = np.arange(0, 20)  # 사건의 발생 횟수

"""
사건의 발생 횟수가 0 ~ 20회 발생할 확률 분포
푸아송 분포로, 각 확률의 합은 1.0이 된다. 

lambda=3.4 일때 약 8, 9회 정도에서 tipping point가 발생한다. 
미분이 0에 근접하는 구간으로 임계점(critical point)이라고도 한다.
"""
plt.xticks(r)  # force the X axis to only print integers.
plt.plot(poisson(3.4), label="lambda=3.4")  # lambda = 3.4
plt.plot(poisson(1), label="lambda=1")  # lambda = 1
plt.plot(poisson(10), label="lambda=10")  # lambda = 10
plt.legend()
plt.show()
