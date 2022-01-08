import math
from matplotlib import pyplot as plt
import numpy as np

all_distances = [
[9310, 10460],
[9350, 10120],
[9490, 9950],
[9650, 9680],
[9890, 9470],
[10200, 9350],
[10430, 9310],
[11010, 9270],
[11380, 9320],
[11800, 9320],
[12170, 9290],
[12270, 9020],
[12300, 8760],
[12710, 8540],
[13050, 8660],
[13440, 8900],
[13770, 9100],
[13880, 9090],
[13920, 9090],
[13860, 9070],
[13730, 9020],
[13550, 8980],
[13390, 8970],
[13250, 9050],
[12970, 9030],
[12650, 8990],
[12280, 8950],
[11930, 9010],
[11610, 9060],
[11090, 8960],
[10590, 8890],
[10120, 8900],
[9810, 8900],
[9440, 9000],
[9210, 9180],
[9100, 9480],
[8930, 9770],
[8900, 10050],
[9090, 10680],
[9150, 11240],
[8970, 11530],
[8940, 11900],
[9030, 12340],
[9100, 12700],
[8960, 12900],
[8960, 13170],
[8990, 13400],
[9000, 13590],
[8980, 13710],
[9000, 13800],
[8960, 13790],
[9020, 13850],
[9000, 13790],
[9070, 13760],
[9040, 13600],
[9100, 13490],
[9170, 13350],
[9120, 13030],
[9080, 12710],
[9090, 12400],
[9160, 12080],
[9140, 11670],
[9160, 11230],
[9200, 10950]
]

p_dist = 4835


def get_cos(dists):
    """from three known distances return cos(angle) in radians"""
    a = dists[1]
    b = dists[0]
    c = p_dist
    return ((b ** 2) + (c ** 2) - (a ** 2)) / (2 * b * c)


def get_point(cos_ang, dists):
    """return [x, y] from angle and hypotenuse"""
    print(cos_ang, dists)
    return [
        cos_ang * dists[0],
        math.sin(math.acos(cos_ang)) * dists[0]
    ]

def get_points():
    """return all points"""
    return np.array([get_point(get_cos(dists), dists) for dists in all_distances])


def display_points():
    """graph points"""

    points = raw_points.copy()
    points[start_flip:end_flip, 1] = -points[start_flip:end_flip, 1]
    plt.scatter(points[:, 0], points[:, 1])

    for i, pt in enumerate(points):
        plt.text(pt[0], pt[1], str(i), ha='center', va='center')

raw_points = get_points()
start_flip, end_flip = 18, 51
display_points()
