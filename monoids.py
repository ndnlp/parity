import sympy
import itertools

def boolmatrix(m):
    return m.applyfunc(lambda x: min(x, 1))

# (aa)*
mu_evenlength = {
    '1': sympy.ImmutableMatrix([[0,1],
                       [1,0]]),
}

mu_parity = {
    '0': sympy.ImmutableMatrix([[1,0],
                       [0,1]]),
    '1': sympy.ImmutableMatrix([[0,1],
                       [1,0]]),
}

# .*ab*a.*
mu1 = {
    'a': sympy.ImmutableMatrix([[1,1,0],
                       [0,1,1],
                       [0,0,1]]),
    'b': sympy.ImmutableMatrix([[1,0,0],
                       [0,1,0],
                       [0,0,1]]),
    'c': sympy.ImmutableMatrix([[1,0,0],
                       [0,0,0],
                       [0,0,1]]),
}

# .*aa.*
mu_aa = {
    'a': sympy.ImmutableMatrix([
        [1,1,0],
        [0,0,1],
        [0,0,1]]),
    'b': sympy.ImmutableMatrix([
        [1,0,0],
        [0,0,0],
        [0,0,1]]),
}

def monoid(mu):
    """Compute the syntactic monoid of automaton mu."""
    m, n = list(mu.values())[0].shape
    assert m == n
    result = {sympy.ImmutableMatrix(sympy.eye(n))}
    changed = True
    while changed:
        changed = False
        for m in list(result):
            for a in mu:
                ma = boolmatrix(m @ mu[a])
                if ma not in result:
                    result.add(ma)
                    changed = True
    return result

def stable_monoid(mu):
    m, n = list(mu.values())[0].shape
    assert m == n
    g = set(mu.values())
    k = 1
    while True:
        g2 = {boolmatrix(x @ y) for x in g for y in g}
        if g == g2:
            return g | {sympy.ImmutableMatrix(sympy.eye(n))}
        g = {boolmatrix(x @ y) for x in g for y in mu.values()}
        k += 1
        
def aperiodic(m):
    for n in range(1, len(m)+1):
        flag = True
        for x in m:
            if boolmatrix(x**n) != boolmatrix(x**(n+1)):
                flag  = False
        if flag: break
    return flag

def da(m):
    for x in m:
        for y in m:
            xy = boolmatrix(x @ y)
            xy_omega = xy
            while True:
                if xy_omega == boolmatrix(xy_omega @ xy_omega):
                    break
                xy_omega = boolmatrix(xy_omega @ xy)
            if xy_omega != boolmatrix(xy_omega @ x @ xy_omega):
                #print(xy_omega, boolmatrix(xy_omega @ x @ xy_omega))
                return False
    return True

for mu in [mu_aa, mu_aba]:
    print("---")
    print('FO', aperiodic(monoid(mu)))
    print('FO[MOD]', aperiodic(stable_monoid(mu)))
    #print(monoid(mu))
    print('FO2', da(monoid(mu)))
    #print(stable_monoid(mu))
    print('FO2[MOD]', da(stable_monoid(mu)))
