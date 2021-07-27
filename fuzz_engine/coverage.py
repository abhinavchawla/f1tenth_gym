import numpy as np


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
