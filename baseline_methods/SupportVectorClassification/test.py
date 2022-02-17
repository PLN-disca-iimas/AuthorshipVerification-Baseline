import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
import utils.split
print(str(p))