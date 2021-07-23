import pickle

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull

from fuzz_engine.cps_fuzz_tester import TreeNode
from fuzz_engine.fuzz_test_gym import F110GymSim


def load_leaves(root):
    if not root.children and root.status == 'ok':
        return [root.obs]
    leaves = []
    for node in root.children.values():
        leaves += load_leaves(node)
    return leaves


with open("root.pkl", "rb") as f:
    TreeNode.sim_state_class = F110GymSim
    root = TreeNode(TreeNode.sim_state_class())
    root = pickle.load(f)
    leaves = load_leaves(root)
    print(leaves)
    vor = Voronoi(leaves)
    print(vor.vertices)

    fig = voronoi_plot_2d(vor)
    plt.xlim(-5, 30)
    plt.ylim(-60, 60)
    plt.show()
    vol = np.zeros(vor.npoints)
    for i, reg_num in enumerate(vor.point_region):
        indices = vor.regions[reg_num]
        if -1 in indices:  # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(vor.vertices[indices]).volume
        print(leaves[i], vol[i])

