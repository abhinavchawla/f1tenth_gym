import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from shapely.geometry import MultiPoint, Point, Polygon


def find_minimum_node(root, ind):
    if not root.children:
        return root.obs[ind]

    else:
        min_node = root.obs[ind]
        for child_node in root.children.values():
            min_node = min(min_node, find_minimum_node(child_node, ind))
        return min_node


def find_maximum_node(root, ind):
    if not root.children:
        return root.obs[ind]

    else:
        max_node = root.obs[ind]
        for child_node in root.children.values():
            max_node = max(max_node, find_maximum_node(child_node, ind))
        return max_node


def find_height_of_tree(root):
    if not root.children:
        return 0

    else:
        depth = -np.inf
        # Compute the depth of each subtree
        for child_node in root.children.values():
            depth = max(depth, find_height_of_tree(child_node))
        return depth + 1


def find_max_distance(root, i):
    return find_maximum_node(root, i) - find_minimum_node(root, i)


def total_crashes(root):
    if not root.children and root.status != 'ok':
        return 1

    else:
        crashes = 0
        for child_node in root.children.values():
            crashes += total_crashes(child_node)
        return crashes


def total_nodes(root):
    if not root.children and root.status != 'ok':
        return 1

    else:
        nodes = 1
        for child_node in root.children.values():
            nodes += total_nodes(child_node)
        return nodes


def find_voronoi_std_dev(root):
    limits_box = [[0, 100], [-5, 5]]
    leaves = np.array(load_leaves(root, limits_box))
    try:
        vor = Voronoi(leaves)

        regions, vertices = voronoi_finite_polygons_2d(vor, 1)
        mask = Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        new_vertices = []
        voronoi_areas = []
        for region in regions:
            polygon = vertices[region]
            shape = list(polygon.shape)
            shape[0] += 1
            p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
            poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
            voronoi_areas.append(ConvexHull(poly).volume)
            new_vertices.append(poly)
        return np.std(np.array(voronoi_areas))
    except:
        return 0


def load_leaves(root, limits_box):
    if not root.children and root.status == 'ok':
        leaf = [normalize_point(limits_box, root.obs)]
        return leaf
    leaves = []
    for node in root.children.values():
        leaves += load_leaves(node, limits_box)
    return leaves


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
        radius = vor.points.ptp().max() * 2

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

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n

            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)
