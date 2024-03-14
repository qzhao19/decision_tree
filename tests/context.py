import os
import sys

if __package__:
    from .. import decision_tree
else:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import decision_tree