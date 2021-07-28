import pickle
import sys

with open("root_rrt0.pkl", "rb") as f:
    root = pickle.load(f)
    print(sys.getsizeof(root))
