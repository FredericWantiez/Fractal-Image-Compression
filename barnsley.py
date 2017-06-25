import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
import pandas as pd


def transformation(args):
    A = np.array(args[:4]).reshape([2, 2])
    B = np.array(args[4:-1])
    p = args[-1]
    return A, B, p


def create_transformations(mat):
    f = list()
    for k in range(len(mat)):
        f.append(transformation(mat[k]))
    return f


def read_mat(filename):
    data = pd.read_table(filename, header=None).values
    tmp = [tab[0].split(' ') for tab in data]
    mat = list()
    for tab in tmp:
        _ = list()
        for elem in tab:
            if elem != "":
                _.append(float(elem))
        mat.append(_)
    return np.array(mat)


mat = read_mat("1.txt")


def curve_fern(n, mat):
    points = list()
    points.append(np.array([0.5, 0]))
    f1, f2, f3, f4 = create_transformations(mat)
    p1, p2, p3, p4 = f1[-1], f2[-1], f3[-1], f4[-1]
    for k in range(n):
        rand = np.random.rand()
        tmp = points[k]
        if rand <= p1:
            tmp = np.dot(f1[0], points[k].T) + f1[1]
        elif rand <= p1 + p2:
            tmp = np.dot(f2[0], points[k].T) + f2[1]
        elif rand <= p1 + p2 + p3:
            tmp = np.dot(f3[0], points[k].T) + f3[1]
        else:
            tmp = np.dot(f4[0], points[k].T) + f4[1]
        points.append(tmp)
    return np.array(points)


def plot_curve(n, mat, save=True, name="Img/fern.png"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    points = curve_fern(n, mat)
    x, y = points[:, 0], points[:, 1]
    ax.set_aspect("equal")
    ax.use_sticky_edges = False
    ax.margins(0.1)
    ax.scatter(x, y, s=0.05, color="g")
    if save:
        plt.savefig("name")
    plt.show()


mat = read_mat("3.txt")
plot_curve(100000, mat)
