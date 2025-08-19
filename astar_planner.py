import numpy as np
import json
import heapq
from typing import List, Tuple, Optional
import os


class AStarPlanner:    
    def __init__(self, map_data_path: str = "calib-results/planning_maps.json"):
        self.load_maps(map_data_path)
        
    def load_maps(self, map_data_path: str):
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
        
    def world_to_grid(self, x: float, z: float) -> Tuple[int, int]:
        col = int((x - self.x_min) / self.cell_size)
        row = int((z - self.z_min) / self.cell_size)
        return row, col
    
    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = self.x_min + col * self.cell_size + self.cell_size / 2
        z = self.z_min + row * self.cell_size + self.cell_size / 2
        return x, z
    
    def is_valid(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width
    
    def is_free(self, row: int, col: int, use_eroded: bool = True) -> bool:
        if not self.is_valid(row, col):
            return False
        
        map_to_use = self.eroded_map if use_eroded else self.planning_map
        return map_to_use[row, col] == 0  # 0 = free
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int, float]]:
        neighbors = []
        
        # 8-connected grid (including diagonals)
        directions = [
            (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
            (0, -1, 1.0),                   (0, 1, 1.0),
            (1, -1, 1.414),  (1, 0, 1.0),  (1, 1, 1.414)
        ]
        
        for dr, dc, cost in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_free(new_row, new_col):
                neighbors.append((new_row, new_col, cost))
        
        return neighbors
    
    def heuristic(self, row1: int, col1: int, row2: int, col2: int) -> float:
        return np.sqrt((row1 - row2)**2 + (col1 - col2)**2)
    
    def plan_path(self, start_x: float, start_z: float, goal_x: float, goal_z: float, 
                  use_eroded: bool = True) -> Optional[List[Tuple[float, float]]]:
        # convert to grid coordinates
        start_row, start_col = self.world_to_grid(start_x, start_z)
        goal_row, goal_col = self.world_to_grid(goal_x, goal_z)
        
        print(f"Planning path from ({start_x:.2f}, {start_z:.2f}) to ({goal_x:.2f}, {goal_z:.2f})")
        print(f"Grid coordinates: ({start_row}, {start_col}) to ({goal_row}, {goal_col})")
        
        # check if start and goal are valid
        if not self.is_free(start_row, start_col, use_eroded):
            print(f"Start position ({start_row}, {start_col}) is not free!")
            return None
            
        if not self.is_free(goal_row, goal_col, use_eroded):
            print(f"Goal position ({goal_row}, {goal_col}) is not free!")
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
            
            # check if we reached the goal
            if current_row == goal_row and current_col == goal_col:
                print(f"Path found! Visited {len(visited)} nodes")
                return self._reconstruct_path(came_from, current_row, current_col)
            
            # explore neighbors
            for neighbor_row, neighbor_col, move_cost in self.get_neighbors(current_row, current_col):
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
    
    def _reconstruct_path(self, came_from: dict, goal_row: int, goal_col: int) -> List[Tuple[float, float]]:
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
    
    def get_map_info(self) -> dict:
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
        
        path = planner.plan_path(start_x, start_z, goal_x, goal_z)
        
        if path:
            print("\nPath found:")
            for i, (x, z) in enumerate(path):
                print(f"  {i}: ({x:.2f}, {z:.2f})")
        else:
            print("No path found")
            
    except Exception as e:
        print(f"Error testing planner: {e}")


if __name__ == "__main__":
    test_planner() 