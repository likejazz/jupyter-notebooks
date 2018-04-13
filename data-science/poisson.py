# %%
import numpy as np
import scipy

import matplotlib.pyplot as plt

l = 3.4  # 사건의 발생 비율(lambda)
r = np.arange(0, 20)  # 사건의 발생 횟수

# 푸아송 분포(poisson distribution)
p = (np.e ** -l * l ** r) / scipy.special.factorial(r)

"""
사건의 발생 횟수가 0 ~ 20회 발생할 확률 분포
약 8, 9회 정도에서 tipping point가 발생한다. 미분이 0에 근접하는 구간으로 임계점(critical point)이라고도 한다.
"""
# force the X axis to only print integers.
plt.xticks(r)
plt.plot(p)
plt.show()
