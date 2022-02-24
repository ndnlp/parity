import argparse
import collections
import sys

ap = argparse.ArgumentParser()
ap.add_argument('-k', type=int, dest="key", default=1)
ap.add_argument('-v', type=int, dest="val", default=2)
args = ap.parse_args()

s = collections.defaultdict(float)
ss = collections.defaultdict(float)
n = collections.Counter()
for line in sys.stdin:
    fields = line.split()
    key = float(fields[args.key-1])
    val = float(fields[args.val-1])
    s[key] += val
    ss[key] += val**2
    n[key] += 1

for key in sorted(s):
    print(key, s[key]/n[key], (ss[key]/n[key] - (s[key]/n[key])**2)**0.5)
