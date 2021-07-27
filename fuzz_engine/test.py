import pickle

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull

from fuzz_engine.cps_fuzz_tester import TreeNode
from fuzz_engine.fuzz_test_gym import F110GymSim


def load_leaves(root, limits_box):
    if not root.children and root.status == 'ok':
        return [normalize_point(limits_box, root.obs)]
    leaves = []
    for node in root.children.values():
        leaves += load_leaves(node, limits_box)
    return leaves


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    print(all_ridges)
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            print(far_point)

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def normalize_point(limits_box, p):
    'distance between two points'

    xscale = 1
    yscale = 1

    if limits_box:
        xscale = limits_box[0][1] - limits_box[0][0]
        yscale = limits_box[1][1] - limits_box[1][0]

    dx = (p[0] - limits_box[0][0]) / xscale
    dy = (p[1] - limits_box[1][0]) / yscale

    return [dx, dy]


with open("root.pkl", "rb") as f:
    TreeNode.sim_state_class = F110GymSim
    root = TreeNode(TreeNode.sim_state_class())
    root = pickle.load(f)
    limits_box = [[0, 100], [-5, 5]]
    leaves = np.array(load_leaves(root, limits_box))
    # print(leaves)
    vor = Voronoi(leaves)
    # print(vor.vertices)

    # fig = voronoi_plot_2d(vor)
    # plt.ylim(0, 1)
    # plt.xlim(0, 1)
    # plt.show()
    # vol = np.zeros(vor.npoints)
    # print(voronoi_finite_polygons_2d(vor))
    #
    #
    # for i, reg_num in enumerate(vor.point_region):
    #     indices = vor.regions[reg_num]
    #     if -1 in indices:  # some regions can be opened
    #         print (indices)
    #         vol[i] = np.inf
    #     else:
    #         vol[i] = ConvexHull(vor.vertices[indices]).volume
    #     # print(leaves[i], vol[i])
    # vol.sort()
    # # print(vol)
    #

# plot
regions, vertices = voronoi_finite_polygons_2d(vor, 1)
print( "--")
print (regions)
print ("--")
print (vertices)

# colorize
for region in regions:
    polygon = vertices[region]
    plt.fill(*zip(*polygon), alpha=0.4)
print(type(leaves))
plt.plot(leaves[:,0], leaves[:,1], 'ko')
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.show()