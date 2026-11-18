"""Uniform-grid spatial index over segment bounding boxes.

Exact (no false negatives): every segment is registered in all cells its bbox touches;
a query for radius r inspects all cells overlapping the square [x-r,x+r]x[y-r,y+r]. Any
segment within r of the point intersects that square, so it shares a cell with it.
"""
from collections import defaultdict

import numpy as np


class GridIndex:
    def __init__(self, ax, ay, bx, by, cell_m=100.0):
        self.cell = float(cell_m)
        minx, maxx = np.minimum(ax, bx), np.maximum(ax, bx)
        miny, maxy = np.minimum(ay, by), np.maximum(ay, by)
        self.x0, self.y0 = float(minx.min()), float(miny.min())
        c = self.cell
        cx0 = np.floor((minx - self.x0) / c).astype(int)
        cx1 = np.floor((maxx - self.x0) / c).astype(int)
        cy0 = np.floor((miny - self.y0) / c).astype(int)
        cy1 = np.floor((maxy - self.y0) / c).astype(int)
        cells = defaultdict(list)
        for i in range(len(ax)):
            for cx in range(cx0[i], cx1[i] + 1):
                for cy in range(cy0[i], cy1[i] + 1):
                    cells[(cx, cy)].append(i)
        self.cells = {k: np.asarray(v, dtype=np.int64) for k, v in cells.items()}

    def query(self, x, y, r):
        c = self.cell
        i0, i1 = int(np.floor((x - r - self.x0) / c)), int(np.floor((x + r - self.x0) / c))
        j0, j1 = int(np.floor((y - r - self.y0) / c)), int(np.floor((y + r - self.y0) / c))
        parts = [self.cells[(i, j)] for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)
                 if (i, j) in self.cells]
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(parts))
