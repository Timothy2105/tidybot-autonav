import numpy as np
import argparse
import os
from pathlib import Path
from plyfile import PlyData
import open3d as o3d
import matplotlib.pyplot as plt
import cv2


# make 2D heightmap from point cloud
def heightmap_max(points_xyz, cell_size=0.10, xlim=None, zlim=None,
                  height_axis=1, height_clip=None):
    pts = points_xyz.astype(np.float64, copy=False)
    print(f"Processing {len(pts)} points...")

    # grid on X–Z, height on Y
    X, Z = pts[:,0], pts[:,2]
    H = pts[:,height_axis]

    # clip to ignore ceiling/outliers
    if height_clip is not None:
        hmin, hmax = height_clip
        m = (H >= hmin) & (H <= hmax)
        X, Z, H = X[m], Z[m], H[m]
        print(f"After height clipping [{hmin:.2f}, {hmax:.2f}]: {len(X)} points")

    if len(X) == 0:
        raise ValueError("No points remaining after filtering")

    # bounds
    xmin, xmax = (np.min(X) if xlim is None else xlim[0],
                  np.max(X) if xlim is None else xlim[1])
    zmin, zmax = (np.min(Z) if zlim is None else zlim[0],
                  np.max(Z) if zlim is None else zlim[1])

    print(f"Map bounds: X[{xmin:.2f}, {xmax:.2f}], Z[{zmin:.2f}, {zmax:.2f}]")

    xmin -= 1e-9; zmin -= 1e-9
    nx = int(np.ceil((xmax - xmin) / cell_size))
    nz = int(np.ceil((zmax - zmin) / cell_size))
    if nx <= 0 or nz <= 0:
        raise ValueError("Invalid grid bounds")

    print(f"Grid size: {nx} x {nz} cells ({cell_size*100:.1f}cm each)")

    ix = np.floor((X - xmin) / cell_size).astype(np.int64)
    iz = np.floor((Z - zmin) / cell_size).astype(np.int64)

    keep = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
    ix, iz, H = ix[keep], iz[keep], H[keep]

    print(f"Points inside grid: {len(ix)}")

    flat = iz * nx + ix
    out = np.full(nx * nz, -np.inf)
    np.maximum.at(out, flat, H)
    out = out.reshape(nz, nx)
    out[~np.isfinite(out)] = np.nan

    x_edges = xmin + np.arange(nx + 1) * cell_size
    z_edges = zmin + np.arange(nz + 1) * cell_size
    
    occupied_cells = np.sum(~np.isnan(out))
    print(f"Occupied cells: {occupied_cells}/{nx*nz} ({100*occupied_cells/(nx*nz):.1f}%)")
    
    return out, x_edges, z_edges


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


def visualize_maps(heightmap, obstacle_map, x_edges, z_edges, save_path=None):
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
    ax2.set_title('Obstacle Map')
    
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
    parser.add_argument("--height_threshold", "-t", type=float, default=0.05,
                       help="Height threshold for obstacles in meters (default: 0.05)")
    parser.add_argument("--height_clip", nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help="Clip height values to range [MIN, MAX] to remove outliers (e.g., 0.0 0.5 for obstacles near floor)")
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
    
    # load point cloud
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    if args.use_open3d:
        points = load_xyz_from_ply_open3d(input_path)
    else:
        points = load_xyz_from_ply(input_path)
    
    # create heightmap
    heightmap, x_edges, z_edges = heightmap_max(
        points, 
        cell_size=args.cell_size,
        xlim=args.xlim,
        zlim=args.zlim,
        height_axis=args.height_axis,
        height_clip=args.height_clip
    )
    
    # create obstacle map
    obstacle_map = create_obstacle_map(heightmap, height_threshold=args.height_threshold)
    
    # save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_heightmap_as_image(heightmap, output_path)
        
        # save obstacle map
        obstacle_output = output_path.with_name(output_path.stem + "_obstacles" + output_path.suffix)
        try:
            import cv2
            obstacle_img = np.full(obstacle_map.shape, 128, dtype=np.uint8)
            obstacle_img[obstacle_map == 0] = 255
            obstacle_img[obstacle_map == 1] = 0
            obstacle_img = np.flipud(obstacle_img)
            cv2.imwrite(str(obstacle_output), obstacle_img)
            print(f"Obstacle map saved: {obstacle_output}")
        except ImportError:
            print("OpenCV not available, obstacle map not saved as image")
    
    # visualize
    if args.visualize:
        viz_output = None
        if args.output:
            viz_output = Path(args.output).with_name(Path(args.output).stem + "_visualization.png")
        visualize_maps(heightmap, obstacle_map, x_edges, z_edges, viz_output)
    
    print("done!")


if __name__ == "__main__":
    main() 