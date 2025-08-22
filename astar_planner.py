import numpy as np
import json
import heapq
import os


class AStarPlanner:    
    def __init__(self, map_data_path="calib-results/planning_maps.json"):
        self.load_maps(map_data_path)
        
    def load_maps(self, map_data_path):
        if not os.path.exists(map_data_path):
            raise FileNotFoundError(f"Planning maps not found: {map_data_path}")
            
        with open(map_data_path, 'r') as f:
            data = json.load(f)
        
        self.grid_info = data['grid_info']
        self.heightmap = np.array(data['heightmap'])
        self.planning_map = np.array(data['planning_map'])
        self.eroded_map = np.array(data['eroded_map'])
        
        self.heightmap[self.heightmap == -999.0] = np.nan
        
        self.height = self.grid_info['height']
        self.width = self.grid_info['width']
        self.cell_size = self.grid_info['cell_size']
        self.x_min = self.grid_info['x_min']
        self.z_min = self.grid_info['z_min']
        
        print(f"Loaded planning maps: {self.height}x{self.width} grid, {self.cell_size}m cells")
        
    def world_to_grid(self, x, z):
        col = int((x - self.x_min) / self.cell_size)
        row = int((z - self.z_min) / self.cell_size)
        return row, col
    
    def grid_to_world(self, row, col):
        x = self.x_min + col * self.cell_size + self.cell_size / 2
        z = self.z_min + row * self.cell_size + self.cell_size / 2
        return x, z
    
    def is_valid(self, row, col):
        return 0 <= row < self.height and 0 <= col < self.width
    
    def is_free(self, row, col, use_eroded=True):
        if not self.is_valid(row, col):
            return False
        
        map_to_use = self.eroded_map if use_eroded else self.planning_map
        return map_to_use[row, col] == 0  # 0 = free
    
    # check if enough clearance for robot movement
    def has_clearance(self, row, col, clearance_radius=2, use_eroded=True):
        map_to_use = self.eroded_map if use_eroded else self.planning_map
        
        # check all cells in 5x5 clearance area
        for dr in range(-clearance_radius, clearance_radius + 1):
            for dc in range(-clearance_radius, clearance_radius + 1):
                check_row = row + dr
                check_col = col + dc
                
                if not self.is_valid(check_row, check_col):
                    return False
                if map_to_use[check_row, check_col] != 0:
                    return False
        
        return True
    
    def get_neighbors(self, row, col, use_clearance=True, use_eroded=True):
        neighbors = []
        
        # 8-connected grid
        directions = [
            (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
            (0, -1, 1.0),                   (0, 1, 1.0),
            (1, -1, 1.414),  (1, 0, 1.0),  (1, 1, 1.414)
        ]
        
        for dr, dc, cost in directions:
            new_row, new_col = row + dr, col + dc
            
            # check if position has 5x5 clearance
            if use_clearance:
                if self.has_clearance(new_row, new_col, clearance_radius=2, use_eroded=use_eroded):
                    neighbors.append((new_row, new_col, cost))
            else:
                if self.is_free(new_row, new_col, use_eroded=use_eroded):
                    neighbors.append((new_row, new_col, cost))
        
        return neighbors
    
    def heuristic(self, row1, col1, row2, col2):
        return np.sqrt((row1 - row2)**2 + (col1 - col2)**2)
    
    # Bresenham's line algorithm
    def _bresenham_line_cells(self, r0, c0, r1, c1):
        dr = abs(r1 - r0); dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = (dr if dr > dc else -dc) // 2
        r, c = r0, c0
        while True:
            yield r, c
            if r == r1 and c == c1:
                break
            e2 = err
            if e2 > -dr:
                err -= dc
                r += sr
            if e2 < dc:
                err += dr
                c += sc

    # to simplify path movements
    def _line_of_sight_free(self, r0, c0, r1, c1, use_eroded=True):
        for rr, cc in self._bresenham_line_cells(r0, c0, r1, c1):
            if not self.has_clearance(rr, cc, clearance_radius=2, use_eroded=use_eroded):
                return False
        return True

    def _reconstruct_path_grid(self, came_from, goal_row, goal_col):
        path = []
        current = (goal_row, goal_col)
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        path.reverse()
        return path

    # merge consecutive steps
    def _compress_collinear(self, path_grid):
        if len(path_grid) <= 2:
            return path_grid[:]
        out = [path_grid[0]]
        prev = path_grid[0]
        dprev = None
        for cur in path_grid[1:]:
            dr = cur[0] - prev[0]
            dc = cur[1] - prev[1]
            # normalize to {-1,0,1}
            dr = 0 if dr == 0 else (1 if dr > 0 else -1)
            dc = 0 if dc == 0 else (1 if dc > 0 else -1)
            d = (dr, dc)
            if d != dprev:
                out.append(cur)  # start new run
                dprev = d
            else:
                out[-1] = cur     # extend current run's end
            prev = cur
        return out

    # greedy line of sight smoothing
    def _string_pull(self, path_grid, use_eroded=True):
        if len(path_grid) <= 2:
            return path_grid[:]
        smoothed = [path_grid[0]]
        anchor_idx = 0
        while True:
            far = anchor_idx + 1
            last_good = far
            while far < len(path_grid):
                a = path_grid[anchor_idx]
                b = path_grid[far]
                if self._line_of_sight_free(a[0], a[1], b[0], b[1], use_eroded=use_eroded):
                    last_good = far
                    far += 1
                else:
                    break
            smoothed.append(path_grid[last_good])
            if last_good == len(path_grid) - 1:
                break
            anchor_idx = last_good
        return smoothed

    def _grid_to_world_path(self, path_grid):
        return [self.grid_to_world(r, c) for (r, c) in path_grid]
    
    def find_nearest_clear_cell(self, start_row, start_col, max_radius=50, use_eroded=True):
        from collections import deque
        grid = self.eroded_map if use_eroded else self.planning_map
        if not self.is_valid(start_row, start_col):
            return None

        q = deque()
        q.append((start_row, start_col))
        seen = set([(start_row, start_col)])

        # 8-connected
        dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        while q:
            r, c = q.popleft()

            # stop if cell has full clearance
            if self.has_clearance(r, c, clearance_radius=2, use_eroded=use_eroded):
                return (r, c)

            # radius guard
            if abs(r - start_row) + abs(c - start_col) > max_radius:
                continue

            # expand over free cells only
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                if (rr, cc) in seen:
                    continue
                if not self.is_valid(rr, cc):
                    continue
                if grid[rr, cc] != 0:
                    continue
                seen.add((rr, cc))
                q.append((rr, cc))
        return None
    
    def plan_path(self, start_x, start_z, goal_x, goal_z, use_eroded=True, simplify="string_pull", skip_start_clearance=False, use_clearance=True, require_goal_clearance=True):
        # convert to grid coordinates
        start_row, start_col = self.world_to_grid(start_x, start_z)
        goal_row, goal_col = self.world_to_grid(goal_x, goal_z)
        
        print(f"Planning path from ({start_x:.2f}, {start_z:.2f}) to ({goal_x:.2f}, {goal_z:.2f})")
        print(f"Grid coordinates: ({start_row}, {start_col}) to ({goal_row}, {goal_col})")
        
        # start clearance
        if not skip_start_clearance and use_clearance:
            if not self.has_clearance(start_row, start_col, clearance_radius=2, use_eroded=use_eroded):
                print(f"Start position ({start_row}, {start_col}) lacks clearance")
                return None
        else:
            print("Skipping start clearance check or using no-clearance policy for neighbors")
            
        # goal clearance
        if require_goal_clearance and use_clearance:
            if not self.has_clearance(goal_row, goal_col, clearance_radius=2, use_eroded=use_eroded):
                print(f"Goal position ({goal_row}, {goal_col}) lacks clearance")
                return None
        else:
            if not self.is_free(goal_row, goal_col, use_eroded=use_eroded):
                print(f"Goal cell is not free")
                return None
        
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_row, start_col))
        
        came_from = {}
        g_score = {(start_row, start_col): 0}
        f_score = {(start_row, start_col): self.heuristic(start_row, start_col, goal_row, goal_col)}
        
        visited = set()
        
        while open_set:
            current_f, current_row, current_col = heapq.heappop(open_set)
            
            if (current_row, current_col) in visited:
                continue
                
            visited.add((current_row, current_col))
            
            # reached goal?
            if current_row == goal_row and current_col == goal_col:
                print(f"Path found! Visited {len(visited)} nodes")
                grid_path = self._reconstruct_path_grid(came_from, current_row, current_col)

                # path simplification
                if simplify in ("rle", "collinear", "dir"):
                    grid_path = self._compress_collinear(grid_path)
                elif simplify in ("string_pull", "los", "smooth", "both"):
                    grid_path = self._compress_collinear(grid_path)
                    grid_path = self._string_pull(grid_path, use_eroded=use_eroded)

                world_path = self._grid_to_world_path(grid_path)
                print(f"Path length: {len(world_path)} waypoints (after simplify='{simplify}')")
                return world_path
            
            # explore neighbors
            for neighbor_row, neighbor_col, move_cost in self.get_neighbors(current_row, current_col, use_clearance=use_clearance, use_eroded=use_eroded):
                if (neighbor_row, neighbor_col) in visited:
                    continue
                
                tentative_g = g_score[(current_row, current_col)] + move_cost
                
                if (neighbor_row, neighbor_col) not in g_score or tentative_g < g_score[(neighbor_row, neighbor_col)]:
                    came_from[(neighbor_row, neighbor_col)] = (current_row, current_col)
                    g_score[(neighbor_row, neighbor_col)] = tentative_g
                    f_score[(neighbor_row, neighbor_col)] = tentative_g + self.heuristic(neighbor_row, neighbor_col, goal_row, goal_col)
                    
                    heapq.heappush(open_set, (f_score[(neighbor_row, neighbor_col)], neighbor_row, neighbor_col))
        
        print("No path found!")
        return None
    
    def plan_safe_path(self, start_x, start_z, goal_x, goal_z, use_eroded=True, simplify="string_pull"):
        sr, sc = self.world_to_grid(start_x, start_z)
        gr, gc = self.world_to_grid(goal_x, goal_z)

        start_clear = self.has_clearance(sr, sc, clearance_radius=2, use_eroded=use_eroded)
        goal_clear  = self.has_clearance(gr, gc, clearance_radius=2, use_eroded=use_eroded)

        if not goal_clear:
            # don't drive into tight/blocked goal
            print("Goal lacks clearance; refusing to plan.")
            return None

        if start_clear:
            # single-stage
            return self.plan_path(start_x, start_z, goal_x, goal_z,
                                  use_eroded=use_eroded, simplify=simplify,
                                  skip_start_clearance=False, use_clearance=True, require_goal_clearance=True)

        # two-stage
        print("Start is obstructed: searching for nearest clear cell...")
        best = self.find_nearest_clear_cell(sr, sc, max_radius=80, use_eroded=use_eroded)
        if best is None:
            print("No clear cell found near the start")
            return None
        clear_x, clear_z = self.grid_to_world(*best)
        print(f"Nearest clear cell: grid {best} -> world ({clear_x:.2f}, {clear_z:.2f})")

        # obstructed -> clear
        leg1 = self.plan_path(start_x, start_z, clear_x, clear_z,
                              use_eroded=use_eroded, simplify="collinear",
                              skip_start_clearance=True, use_clearance=False, require_goal_clearance=True)
        if not leg1 or len(leg1) < 2:
            print("Failed to reach the clear cell from obstructed start")
            return None

        # clear -> goal
        leg2 = self.plan_path(clear_x, clear_z, goal_x, goal_z,
                              use_eroded=use_eroded, simplify="string_pull",
                              skip_start_clearance=False, use_clearance=True, require_goal_clearance=True)
        if not leg2 or len(leg2) < 2:
            print("Failed to plan from clear cell to goal")
            return None

        # merge legs
        merged = leg1 + leg2[1:]

        # final smoothing over eroded map
        grid = [self.world_to_grid(x, z) for (x, z) in merged]
        grid = self._compress_collinear(grid)
        grid = self._string_pull(grid, use_eroded=use_eroded)
        merged_smoothed = self._grid_to_world_path(grid)

        print(f"Two-stage path: {len(merged_smoothed)} waypoints after smoothing")
        return merged_smoothed
    
    def _reconstruct_path(self, came_from, goal_row, goal_col):
        path_grid = []
        current = (goal_row, goal_col)
        
        while current in came_from:
            path_grid.append(current)
            current = came_from[current]
        path_grid.append(current)  # add start
        
        path_grid.reverse()
        
        # convert to world coordinates
        path_world = []
        for row, col in path_grid:
            x, z = self.grid_to_world(row, col)
            path_world.append((x, z))
        
        print(f"Path length: {len(path_world)} waypoints")
        return path_world
    
    def get_map_info(self):
        return {
            'grid_size': (self.height, self.width),
            'cell_size': self.cell_size,
            'world_bounds': {
                'x': (self.x_min, self.x_min + self.width * self.cell_size),
                'z': (self.z_min, self.z_min + self.height * self.cell_size)
            },
            'free_cells_original': int(np.sum(self.planning_map == 0)),
            'obstacle_cells_original': int(np.sum(self.planning_map == 1)),
            'free_cells_eroded': int(np.sum(self.eroded_map == 0)),
            'obstacle_cells_eroded': int(np.sum(self.eroded_map == 1))
        }


def test_planner():
    try:
        planner = AStarPlanner()
        print("Map info:", planner.get_map_info())
        
        start_x, start_z = 0.0, 0.0
        goal_x, goal_z = 2.0, 2.0
        
        # test different simplification methods
        for method in ["none", "collinear", "string_pull"]:
            print(f"\n--- Testing simplify='{method}' ---")
            path = planner.plan_path(start_x, start_z, goal_x, goal_z, simplify=method)
            
            if path:
                print(f"Path found with {len(path)} waypoints:")
                for i, (x, z) in enumerate(path[:10]):  # first 10 waypoints
                    print(f"  {i}: ({x:.2f}, {z:.2f})")
                if len(path) > 10:
                    print(f"  ... and {len(path) - 10} more waypoints")
            else:
                print("No path found")
            
    except Exception as e:
        print(f"Error testing planner: {e}")


if __name__ == "__main__":
    test_planner() 