import numpy as np
import argparse
import os
from pathlib import Path
from plyfile import PlyData
import open3d as o3d
import matplotlib.pyplot as plt
import cv2


# make 2D heightmap from point cloud
def heightmap_and_clearance(
    pts, 
    cell_size=0.10,
    # visualization
    height_axis=1, viz_clip=(-0.1, 2.0),
    # planning
    clear_low=0.20,            
    clear_high=1.50,
    floor_required=True,      
    min_floor_pts=2,        
):
    pts = pts.astype(np.float64, copy=False)
    print(f"Processing {len(pts)} points...")
    
    X, Y, Z = pts[:,0], pts[:,height_axis], pts[:,2]

    # grid bounds
    xmin, xmax = X.min()-1e-9, X.max()
    zmin, zmax = Z.min()-1e-9, Z.max()
    
    print(f"Map bounds: X[{xmin:.2f}, {xmax:.2f}], Z[{zmin:.2f}, {zmax:.2f}]")
    
    nx = int(np.ceil((xmax - xmin)/cell_size))
    nz = int(np.ceil((zmax - zmin)/cell_size))
    
    print(f"Grid size: {nx} x {nz} cells ({cell_size*100:.1f}cm each)")

    ix = np.floor((X - xmin)/cell_size).astype(np.int64)
    iz = np.floor((Z - zmin)/cell_size).astype(np.int64)
    keep = (ix>=0)&(ix<nx)&(iz>=0)&(iz<nz)
    ix, iz, Y = ix[keep], iz[keep], Y[keep]
    
    print(f"Points inside grid: {len(ix)}")

    flat = iz*nx + ix
    ncell = nx*nz

    # visualization height map
    Yviz = Y
    if viz_clip is not None:
        lo, hi = viz_clip
        m = (Yviz>=lo)&(Yviz<=hi)
        Y_for_viz = Yviz[m]
        flat_viz  = flat[m]
        print(f"After viz clipping [{lo:.2f}, {hi:.2f}]: {len(Y_for_viz)} points")
    else:
        Y_for_viz = Y
        flat_viz = flat

    Hflat = np.full(ncell, -np.inf)
    np.maximum.at(Hflat, flat_viz, Y_for_viz)
    H = Hflat.reshape(nz, nx)
    H[~np.isfinite(H)] = np.nan

    # planning map using clearance band
    floor_mask = (Y < clear_low)
    band_mask  = (Y >= clear_low) & (Y <= clear_high)
    above_mask = (Y > clear_high)

    cnt_total = np.zeros(ncell, dtype=np.int32)
    cnt_floor = np.zeros(ncell, dtype=np.int32)
    cnt_band  = np.zeros(ncell, dtype=np.int32)
    cnt_above = np.zeros(ncell, dtype=np.int32)

    np.add.at(cnt_total, flat, 1)
    np.add.at(cnt_floor, flat[floor_mask], 1)
    np.add.at(cnt_band,  flat[band_mask],  1)
    np.add.at(cnt_above, flat[above_mask], 1)

    # decision per cell
    plan = np.full(ncell, -1, dtype=np.int8)
    plan[cnt_band > 0] = 1
    free_mask = (cnt_band == 0)
    if floor_required:
        free_mask &= (cnt_floor >= min_floor_pts) | (cnt_total > 0)
    plan[free_mask] = 0

    plan_map = plan.reshape(nz, nx)
    
    # stats for viz
    occupied_cells_viz = np.sum(~np.isnan(H))
    free_cells = np.sum(plan_map == 0)
    obstacle_cells = np.sum(plan_map == 1)
    unknown_cells = np.sum(plan_map == -1)
    
    print(f"Visualization occupied cells: {occupied_cells_viz}/{nx*nz} ({100*occupied_cells_viz/(nx*nz):.1f}%)")
    print(f"Planning - Free cells: {free_cells}")
    print(f"Planning - Obstacle cells: {obstacle_cells}")  
    print(f"Planning - Unknown cells: {unknown_cells}")
    print(f"Clearance band: [{clear_low:.2f}, {clear_high:.2f}]m")

    # edges for plotting/alignment
    x_edges = xmin + np.arange(nx+1)*cell_size
    z_edges = zmin + np.arange(nz+1)*cell_size
    return H, plan_map, x_edges, z_edges


def load_xyz_from_ply(path):
    print(f"Loading PLY file: {path}")
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    
    # get xyz coords
    points = np.column_stack([vertices['x'], vertices['y'], vertices['z']])
    print(f"Loaded {len(points)} points from PLY file")
    
    return points


# using open3d
def load_xyz_from_ply_open3d(path):
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    print(f"Loaded {len(pts)} points from PLY file using Open3D")
    return pts


def create_obstacle_map(heightmap, height_threshold=0.05, floor_height=None):
    if floor_height is None:
        floor_height = np.nanmin(heightmap)
    
    obstacle_map = np.full_like(heightmap, -1, dtype=np.int8)  # -1 = unknown
    
    # free space 
    valid_mask = ~np.isnan(heightmap)
    free_mask = valid_mask & (heightmap - floor_height < height_threshold)
    obstacle_map[free_mask] = 0  # 0 = free
    
    # obstacles
    obstacle_mask = valid_mask & (heightmap - floor_height >= height_threshold)
    obstacle_map[obstacle_mask] = 1  # 1 = obstacle
    
    print(f"Floor height: {floor_height:.3f}m")
    print(f"Free cells: {np.sum(obstacle_map == 0)}")
    print(f"Obstacle cells: {np.sum(obstacle_map == 1)}")
    print(f"Unknown cells: {np.sum(obstacle_map == -1)}")
    
    return obstacle_map


def visualize_maps(heightmap, obstacle_map, x_edges, z_edges, save_path=None, map_type="Obstacle"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    extent = [x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]]
    
    # heightmap
    im1 = ax1.imshow(heightmap, origin='lower', extent=extent, cmap='viridis')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('Height Map')
    plt.colorbar(im1, ax=ax1, label='Height (m)')
    
    # obstacle map
    obstacle_display = obstacle_map.astype(float)
    obstacle_display[obstacle_map == -1] = 0.5  # gray for unknown
    
    im2 = ax2.imshow(obstacle_display, origin='lower', extent=extent, 
                     cmap='RdYlBu_r', vmin=0, vmax=1)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title(f'{map_type} Map')
    
    # colorbar for obstacle map
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_ticks([0, 0.5, 1])
    cbar2.set_ticklabels(['Free', 'Unknown', 'Obstacle'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


# save heightmap as grayscale img
def save_heightmap_as_image(heightmap, output_path):
    
    # normalize heightmap
    valid_mask = ~np.isnan(heightmap)
    if not np.any(valid_mask):
        print("No valid height data to save")
        return
    
    h_min = np.nanmin(heightmap)
    h_max = np.nanmax(heightmap)
    
    # create image array
    img = np.full(heightmap.shape, 128, dtype=np.uint8)
    img[valid_mask] = ((heightmap[valid_mask] - h_min) / (h_max - h_min) * 255).astype(np.uint8)
    
    # flip vertically for image coordinate system
    img = np.flipud(img)
    
    cv2.imwrite(str(output_path), img)
    print(f"Heightmap saved as image: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create 2D occupancy map from point cloud")
    parser.add_argument("--input", "-i", required=True, help="Input PLY file path")
    parser.add_argument("--output", "-o", help="Output image path (optional)")
    parser.add_argument("--cell_size", "-c", type=float, default=0.10, 
                       help="Grid cell size in meters (default: 0.10 = 10cm)")
    parser.add_argument("--height_threshold", "-t", nargs='*', type=float, 
                       help="Height threshold for obstacles. Single value (e.g., 0.05) for simple threshold, or two values (e.g., 0.2 1.5) for clearance band [low, high]. Default: 0.05 for simple mode")
    parser.add_argument("--viz_clip", nargs=2, type=float, metavar=('MIN', 'MAX'), default=(-0.1, 2.0),
                       help="Height range for visualization (default: -0.1 2.0)")
    parser.add_argument("--xlim", nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help="Limit X range to [MIN, MAX]")
    parser.add_argument("--zlim", nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help="Limit Z range to [MIN, MAX]")
    parser.add_argument("--height_axis", type=int, default=1, choices=[1, 2],
                       help="Which axis is height: 1 for Y-up (default), 2 for Z-up")
    parser.add_argument("--visualize", "-v", action="store_true", 
                       help="Show visualization with matplotlib")
    parser.add_argument("--use_open3d", action="store_true",
                       help="Use Open3D to load PLY file instead of plyfile")
    
    args = parser.parse_args()
    
    # parse height threshold and auto-detect mode
    if args.height_threshold is None:
        # default to simple threshold mode
        simple_threshold = 0.05
        clear_low, clear_high = None, None
        use_clearance = False
    elif len(args.height_threshold) == 1:
        simple_threshold = args.height_threshold[0]
        clear_low, clear_high = None, None
        use_clearance = False
    elif len(args.height_threshold) == 2:
        clear_low, clear_high = args.height_threshold
        simple_threshold = None
        use_clearance = True
    else:
        print("Error: --height_threshold accepts 1 or 2 values")
        return
    
    # load point cloud
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    if args.use_open3d:
        points = load_xyz_from_ply_open3d(input_path)
    else:
        points = load_xyz_from_ply(input_path)
    
    # create heightmap and planning map
    if use_clearance:
        print("Using clearance band planning...")
        heightmap, plan_map, x_edges, z_edges = heightmap_and_clearance(
            points,
            cell_size=args.cell_size,
            height_axis=args.height_axis,
            viz_clip=args.viz_clip,
            clear_low=clear_low,
            clear_high=clear_high
        )
        # use planning map as obstacle map for visualization
        obstacle_map = plan_map
    else:
        print("Using simple height threshold...")
        # simple mode: use viz_clip range, set clearance band to cover full range
        viz_min, viz_max = args.viz_clip
        heightmap, plan_map, x_edges, z_edges = heightmap_and_clearance(
            points,
            cell_size=args.cell_size,
            height_axis=args.height_axis,
            viz_clip=args.viz_clip,
            clear_low=viz_min,
            clear_high=simple_threshold
        )
        # create simple obstacle map from heightmap
        obstacle_map = create_obstacle_map(heightmap, height_threshold=simple_threshold)
    
    # save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_heightmap_as_image(heightmap, output_path)
        
        # save obstacle/planning map
        obstacle_output = output_path.with_name(output_path.stem + "_obstacles" + output_path.suffix)
        try:
            import cv2
            obstacle_img = np.full(obstacle_map.shape, 128, dtype=np.uint8)
            obstacle_img[obstacle_map == 0] = 255
            obstacle_img[obstacle_map == 1] = 0
            obstacle_img = np.flipud(obstacle_img)
            cv2.imwrite(str(obstacle_output), obstacle_img)
            map_type = "planning" if use_clearance else "obstacle"
            print(f"{map_type.capitalize()} map saved: {obstacle_output}")
        except ImportError:
            print("OpenCV not available, obstacle map not saved as image")
    
    # visualize
    if args.visualize:
        viz_output = None
        if args.output:
            viz_output = Path(args.output).with_name(Path(args.output).stem + "_visualization.png")
        map_type = "Planning" if use_clearance else "Obstacle"
        visualize_maps(heightmap, obstacle_map, x_edges, z_edges, viz_output, map_type)
    
    print("done!")


if __name__ == "__main__":
    main() 