import sys
import math

print(r'\documentclass{standalone}')
print(r'\usepackage{tikz,pgfplots}')
print(r'\usepgfplotslibrary{fillbetween}')
print(r'\begin{document}')
print(r'\begin{tikzpicture}')
print(r'  \begin{axis}')

for filename in sys.argv[1:]:

    data = []
    for line in open(filename):
        x, ymean, ystd = line.split()
        x = float(x)
        ymean = float(ymean)
        ystd = float(ystd)
        data.append((x, ymean, ystd))

    print(r'    \addplot coordinates {')
    for x, ymean, ystd in data:
        print(rf'      ({x},{ymean})')
    print(r'    };')

print(r'  \end{axis}')
print(r'\end{tikzpicture}')
print(r'\end{document}')
