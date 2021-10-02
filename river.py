import sys
import math

scale = 1/100/math.log(2)
#scale = 100

data = []
for line in sys.stdin:
    x, ymean, ystd = line.split()
    x = float(x)
    ymean = float(ymean)
    ystd = float(ystd)
    data.append((x, ymean, ystd))

print(r'\documentclass{standalone}')
print(r'\usepackage{tikz,pgfplots}')
print(r'\usepgfplotslibrary{fillbetween}')
print(r'\begin{document}')
print(r'\begin{tikzpicture}')
print(r'  \begin{axis}')

print(r'    \addplot[mark=none] coordinates {')
for x, ymean, ystd in data:
    print(rf'      ({x},{ymean * scale})')
print(r'    };')

print(r'    \addplot[draw=none,mark=none,name path=above] coordinates {')
for x, ymean, ystd in data:
    print(rf'      ({x},{(ymean+ystd) * scale})')
print(r'    };')

print(r'    \addplot[draw=none,mark=none,name path=below] coordinates {')
for x, ymean, ystd in data:
    print(rf'      ({x},{(ymean-ystd) * scale})')
print(r'    };')

print(r'    \addplot fill between[of=above and below];')

print(r'  \end{axis}')
print(r'\end{tikzpicture}')
print(r'\end{document}')

    
