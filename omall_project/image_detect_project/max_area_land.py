# -*- coding: utf-8 -*-
import numpy as np
import copy


def maxAreaOfIsland_v2(grid):
    m = len(grid)
    if m == 0:
        return 0
    n = len(grid[0])
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid, dtype=np.float32)
    save_matrix = []
    for i in range(m):  # column
        save_line = []
        save_part = []
        next_line_connect = False
        for j in range(n+1):  # row
            if j != n and grid[i, j] != 0.0:
                save_part.append((i, j))
                if i != m-1 and grid[i+1, j] != 0.0:
                    save_part.extend([(i+1, j)])
                    next_line_connect = True
            else:
                save_line += [save_part] if save_part != [] else ''
                save_part = []
        save_matrix.append(save_line)
        if not next_line_connect:
            continue
    # flatten
    flat_matrix = []
    for i in save_matrix:
        flat_matrix += i
    # get all position
    result = [set(flat_matrix[0])]
    for i in flat_matrix[1:]:
        new = True
        for j in range(len(result)):
            if set(i) & result[j]:
                result[j] |= set(i)
                new = False
        if new:
            result.append(set(i))
    # solve merge
    for i in range(len(result)-1):
        if result[i] & result[i+1]:
            result[i] |= result[i+1]
            result.remove(result[i+1])
    # get max position
    pa = {}
    max_area = -1
    for figure_size in result:
        max_h = -1
        max_w = -1
        min_h = float('inf')
        min_w = float('inf')
        for h, w in figure_size:
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            min_h = min(min_h, h)
            min_w = min(min_w, w)
        area = (max_h - min_h) * (max_w - min_w)
        pa[area] = ((min_h, min_w), (max_h, max_w))
        max_area = max(max_area, area)
    print('max_area {}, position:{}'.format(max_area, pa[max_area]))
    return max_area, pa[max_area]


def get_rectangle_index(arr):
    arr = np.array(arr)
    return (np.min(arr[:, 0]), np.min(arr[:, 1])), (np.max(arr[:, 0]), np.max(arr[:, 1]))


def maxAreaOfIsland(grid):
    m = len(grid)
    if m == 0:
        return 0
    n = len(grid[0])
    ans = 0

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n:
            return 0, [None]
        if grid[i][j] == 0:
            return 0, [None]
        cur_loc = [[i, j]]
        grid[i][j] = 0
        top, top_loc = dfs(i + 1, j)
        bottom, bottom_loc = dfs(i - 1, j)
        left, left_loc = dfs(i, j - 1)
        right, right_loc = dfs(i, j + 1)
        res_loc = [x for x in cur_loc + top_loc +
                   bottom_loc + left_loc + right_loc if x is not None]
        return 1 + sum([top, bottom, left, right]), res_loc

    res_loc = []
    for i in range(m):
        for j in range(n):
            val, loc = dfs(i, j)
            if loc[0] is not None:
                res_loc.append(loc)
            ans = max(ans, val)
    return ans, res_loc


def get_candidate_rectangle(grid, fuzz_dis=1):
    new_grid = copy.deepcopy(grid)
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0.0:
                flag = False
                if (sum(grid[max(i - fuzz_dis, 0): i, j]) > 0) and (
                        sum(grid[i + 1:min(i + fuzz_dis + 1, len(grid) - 1), j]) > 0):
                    flag = True
                if (sum(grid[i, max(j - fuzz_dis, 0): j]) > 0) and (
                        sum(grid[i, j + 1:min(j + fuzz_dis + 1, len(grid[0]) - 1)]) > 0):
                    flag = True
                if flag:
                    new_grid[i][j] = 1.0
    return new_grid


if __name__ == '__main__':
    arr = [[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
           [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]]
    connected_arr = get_candidate_rectangle(arr)
    print(np.array(connected_arr))
    # --- v2 --- (no recursive)#
    maxAreaOfIsland_v2(connected_arr)
    
    # ---- v1 --- (DFS recursive)#
    # max_area, rectangle_locs = maxAreaOfIsland(connected_arr)
    # for loc in rectangle_locs:
    #     rectangle_xy = get_rectangle_index(loc)
    #     print(rectangle_xy)
