import sys
import importlib
import sitecustomize
print('before', sys.path[:3])
importlib.reload(sitecustomize)
print('after', sys.path[:3])

