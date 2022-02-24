import sys
import math
import collections
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-s', type=int, dest="s")
ap.add_argument('-x', type=int, dest="x", default=1)
ap.add_argument('-y', type=int, dest="y", default=2)
args = ap.parse_args()

class Stats:
    def __init__(self):
        self.s = 0
        self.ss = 0
        self.n = 0
        
    def point(self, y):
        self.s += y
        self.ss += y**2
        self.n += 1

    def mean(self):
        return self.s/self.n

    def var(self):
        return self.ss/self.n - (self.s/self.n)**2
    
    def stdev(self):
        return self.var() ** 0.5

data = collections.defaultdict(lambda: collections.defaultdict(Stats))
for line in sys.stdin:
    fields = line.split()
    if args.s:
        s = int(fields[args.s-1])
    else:
        s = None
    x = float(fields[args.x-1])
    y = float(fields[args.y-1])
    data[s][x].point(y)

print(r'\documentclass{standalone}')
print(r'\usepackage{tikz,pgfplots}')
print(r'\usepgfplotslibrary{fillbetween}')
print(r'\begin{document}')
print(r'\begin{tikzpicture}')
print(r'  \begin{axis}')

for s in sorted(data):
    print(r'    \addplot coordinates {')
    for x in sorted(data[s]):
        ymean = data[s][x].mean()
        print(rf'      ({x},{ymean})')
    print(r'    };')
    if s is not None:
        print(rf'    \addlegendentry{{{s}}}')

print(r'  \end{axis}')
print(r'\end{tikzpicture}')
print(r'\end{document}')
