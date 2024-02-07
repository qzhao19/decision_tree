import os
import sys

if __package__:
    from .. import dtree
else:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import dtree