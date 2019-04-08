# %%
import sympy

sympy.init_printing()
# Jupyter Notebook에서는 아래와 같이 초기화한다.
# sympy.init_printing(use_latex='mathjax')

y = sympy.symbols('y')

# %%
c = (2 - y) ** 2

# %% c 수식의 y에 대한 편미분
sympy.Derivative(c, y)

# %% 편미분 계산
sympy.Derivative(c, y).doit()

# %% y=3 일때 편미분 계산
sympy.Derivative(c, y).doit().subs({y: 3})

# %% 수식 정리
x, y = sympy.symbols('x y')
y = (2 * x - 1) ** 2
sympy.simplify(sympy.Derivative(y).doit())

# %% 미분 계산은 `doit()`과 `diff()`로 한다.
l, z = sympy.symbols('l z')
l = 1 / (1 + sympy.exp(-z))

assert sympy.Derivative(l, z).doit() == sympy.diff(l, z)

# %% 수식의 미분 풀이 과정
w = sympy.symbols('w')
C = 1 / 2 * (14 - 7 * w) ** 2 + 1 / 2 * (15 - 8 * w) ** 2 + 1 / 2 * (22 - 10 * w) ** 2
C

# %%
sympy.Derivative(C)

# %%
sympy.Derivative(C).doit()

# %%
sympy.Eq(sympy.Derivative(C).doit(), 0)

# %%
sympy.solve(sympy.Eq(sympy.Derivative(C).doit(), 0), w)
