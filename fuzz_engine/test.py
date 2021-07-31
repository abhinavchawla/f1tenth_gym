import pickle

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import Voronoi, ConvexHull
from shapely.geometry import Polygon, MultiPoint, Point

from fuzz_engine.cps_fuzz_tester import TreeNode
from fuzz_engine.fuzz_test_gym import F110GymSim

def load_leaves(root, limits_box):
    if not root.children and root.status == 'ok':
        print(root.obs)
        leaf = [normalize_point(limits_box, root.obs)]
        print(leaf)
        return leaf
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

    # print(all_ridges)
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
            print("Vertex: ", vor.vertices[v2])

            print("Direction: ", direction)
            far_point = vor.vertices[v2] + direction * radius
            print("Check1: ", lineRayIntersectionPoint(vor.vertices[v2], direction, np.array([0,0]), np.array([0,1])))
            print("Check2: ", lineRayIntersectionPoint(vor.vertices[v2], direction, np.array([0,1]), np.array([1,1])))
            print("Check3: ", lineRayIntersectionPoint(vor.vertices[v2], direction, np.array([1,0]), np.array([1,1])))
            print("Check4: ", lineRayIntersectionPoint(vor.vertices[v2], direction, np.array([1,0]), np.array([0,0])))

            # print(far_point)

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

def lineRayIntersectionPoint (rayOrigin, rayDirection, point1, point2):
    v1 = rayOrigin - point1
    v2 = point2 - point1
    v3 = np.array([-rayDirection[1], rayDirection[0]])
    t1 = np.cross(v2, v1) / np.dot(v2, v3)
    t2 = np.dot(v1, v3) / np.dot(v2, v3)
    if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
        return [rayOrigin + t1 * rayDirection]
    return []

with open("root_rrt2.pkl", "rb") as f:
    TreeNode.sim_state_class = F110GymSim
    root = TreeNode(TreeNode.sim_state_class())
    root = pickle.load(f)
    limits_box = [[0, 100], [-5, 5]]
    leaves = np.array(load_leaves(root, limits_box))
    vor = Voronoi(leaves)

    regions, vertices = voronoi_finite_polygons_2d(vor, 1)
    pts = MultiPoint([Point(i) for i in leaves])
    mask = Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    new_vertices = []
    sm = 0
    for region in regions:
        polygon = vertices[region]
        shape = list(polygon.shape)
        shape[0] += 1
        p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
        poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
        sm+=ConvexHull(poly).volume
        new_vertices.append(poly)
        plt.fill(*zip(*poly), "brown", alpha=0.4, edgecolor='black')


    plt.plot(leaves[:, 0], leaves[:, 1], 'ko')
    plt.xlim(-1,2)
    plt.ylim(-1,2)

    plt.title("Blast 2620 S3C 5009 P1")
    plt.show()
