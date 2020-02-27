import random as r
import sys

inp = float(sys.argv[1])

if r.randrange(2) > 0:
    print('Up')
    print(inp + r.randrange(3) / 100)
else:
    print('Down')
    print(inp - r.randrange(3) / 100)
