import numpy as np
import cv2
import re
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_calibration_data(calibration_text):
    """Parse all camera poses from the calibration file"""
    cam_positions_raw = re.findall(r"CAMERA ESTIMATED END POSITION \(SLAM\):\s+\[x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\]", calibration_text)
    cam_orientations_raw = re.findall(r"CAMERA ESTIMATED END ORIENTATION \(SLAM\) \(quaternion\):\s+\[w=([-\d.]+), x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\]", calibration_text)

    if len(cam_positions_raw) != len(cam_orientations_raw):
        raise ValueError("Inconsistent number of camera positions and orientations found.")
    if len(cam_positions_raw) < 3:
        raise ValueError("Need at least 3 points to define a plane.")

    # Process camera poses
    positions = []
    orientations = []
    
    for pos_raw, orient_raw in zip(cam_positions_raw, cam_orientations_raw):
        # Position vector
        position = np.array([float(p) for p in pos_raw])
        positions.append(position)
        
        # Quaternion to rotation matrix
        quat_wxyz = [float(q) for q in orient_raw]
        
        # [w, x, y, z] -> [x, y, z, w]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        
        rotation_matrix = R.from_quat(quat_xyzw).as_matrix()
        orientations.append(rotation_matrix)

    return positions, orientations

def calculate_plane_of_best_fit(points):
    """
    Calculate the plane of best fit using PCA
    Returns: normal vector, center point, and plane equation ax + by + cz + d = 0
    """
    # Convert to numpy array
    points = np.array(points)
    
    # Center the points
    center = np.mean(points, axis=0)
    centered_points = points - center
    
    # Use PCA to find the principal components
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    
    # The normal vector is the component with the smallest variance (3rd principal component)
    normal = pca.components_[2]  # Smallest eigenvalue corresponds to normal
    
    # Calculate the d parameter for the plane equation ax + by + cz + d = 0
    d = -np.dot(normal, center)
    
    # Calculate the plane equation coefficients
    a, b, c = normal
    plane_equation = [a, b, c, d]
    
    return normal, center, plane_equation, pca

def calculate_point_to_plane_distances(points, plane_equation):
    """Calculate the distance from each point to the plane"""
    a, b, c, d = plane_equation
    normal = np.array([a, b, c])
    normal_magnitude = np.linalg.norm(normal)
    
    distances = []
    for point in points:
        # Distance formula: |ax + by + cz + d| / sqrt(a² + b² + c²)
        distance = abs(np.dot(normal, point) + d) / normal_magnitude
        distances.append(distance)
    
    return np.array(distances)

def analyze_plane_quality(points, plane_equation, normal, center):
    """Analyze the quality of the plane fit"""
    # Calculate distances from points to plane
    distances = calculate_point_to_plane_distances(points, plane_equation)
    
    # Statistics
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    max_distance = np.max(distances)
    min_distance = np.min(distances)
    
    # Calculate R-squared (coefficient of determination)
    total_variance = np.var(points, axis=0).sum()
    residual_variance = np.var(distances)
    r_squared = 1 - (residual_variance / total_variance) if total_variance > 0 else 0
    
    return {
        'distances': distances,
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'max_distance': max_distance,
        'min_distance': min_distance,
        'r_squared': r_squared
    }

def visualize_plane_and_points(points, normal, center, plane_equation, quality_metrics):
    """Create a 3D visualization of the plane and points"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', marker='o', s=50, label='Camera Positions')
    
    # Create a grid for the plane
    a, b, c, d = plane_equation
    
    # Find the range of x, y coordinates
    x_min, x_max = points[:, 0].min() - 0.5, points[:, 0].max() + 0.5
    y_min, y_max = points[:, 1].min() - 0.5, points[:, 1].max() + 0.5
    
    # Create a meshgrid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    
    # Calculate z values for the plane: z = (-ax - by - d) / c
    if abs(c) > 1e-10:  # Avoid division by zero
        zz = (-a * xx - b * yy - d) / c
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='red', label='Plane of Best Fit')
    
    # Plot the center point
    ax.scatter([center[0]], [center[1]], [center[2]], c='red', marker='*', s=200, label='Center Point')
    
    # Plot the normal vector
    normal_end = center + normal * 0.5  # Scale the normal for visualization
    ax.quiver(center[0], center[1], center[2], 
              normal[0], normal[1], normal[2], 
              color='green', arrow_length_ratio=0.1, label='Normal Vector')
    
    # Add labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(f'Camera Positions and Plane of Best Fit\n'
                f'Mean Distance: {quality_metrics["mean_distance"]:.4f}m, '
                f'Std: {quality_metrics["std_distance"]:.4f}m, '
                f'R²: {quality_metrics["r_squared"]:.4f}')
    
    ax.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('calib-results/plane_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_plane_results(plane_equation, normal, center, quality_metrics, points):
    """Save the plane analysis results to files"""
    import os
    os.makedirs('calib-results', exist_ok=True)
    
    # Save plane equation and analysis
    with open('calib-results/plane_analysis.txt', 'w') as f:
        f.write("PLANE OF BEST FIT ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Plane Equation (ax + by + cz + d = 0):\n")
        a, b, c, d = plane_equation
        f.write(f"  a = {a:.6f}\n")
        f.write(f"  b = {b:.6f}\n")
        f.write(f"  c = {c:.6f}\n")
        f.write(f"  d = {d:.6f}\n")
        f.write(f"  Equation: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0\n\n")
        
        f.write("Normal Vector:\n")
        f.write(f"  [x={normal[0]:.6f}, y={normal[1]:.6f}, z={normal[2]:.6f}]\n")
        f.write(f"  Magnitude: {np.linalg.norm(normal):.6f}\n\n")
        
        f.write("Center Point:\n")
        f.write(f"  [x={center[0]:.6f}, y={center[1]:.6f}, z={center[2]:.6f}]\n\n")
        
        f.write("Quality Metrics:\n")
        f.write(f"  Mean distance to plane: {quality_metrics['mean_distance']:.6f} meters\n")
        f.write(f"  Standard deviation: {quality_metrics['std_distance']:.6f} meters\n")
        f.write(f"  Maximum distance: {quality_metrics['max_distance']:.6f} meters\n")
        f.write(f"  Minimum distance: {quality_metrics['min_distance']:.6f} meters\n")
        f.write(f"  R-squared: {quality_metrics['r_squared']:.6f}\n\n")
        
        f.write("Individual Point Distances:\n")
        for i, (point, distance) in enumerate(zip(points, quality_metrics['distances'])):
            f.write(f"  Point {i+1}: [{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}] -> {distance:.6f}m\n")
    
    # Save plane equation as numpy array for easy loading
    np.save('calib-results/plane_equation.npy', np.array(plane_equation))
    np.save('calib-results/plane_normal.npy', normal)
    np.save('calib-results/plane_center.npy', center)
    
    print(f"Results saved to calib-results/plane_analysis.txt")
    print(f"Plane equation saved to calib-results/plane_equation.npy")

if __name__ == "__main__":
    # Load and parse data
    calibration_file = "calib-results/calib.txt" 
    try:
        with open(calibration_file, 'r') as f:
            calibration_text = f.read()
        positions, orientations = parse_calibration_data(calibration_text)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        exit()

    num_points = len(positions)
    print(f"Successfully parsed {num_points} camera positions from the calibration file.")

    # Calculate plane of best fit
    print("\nCalculating plane of best fit...")
    normal, center, plane_equation, pca = calculate_plane_of_best_fit(positions)
    
    # Analyze plane quality
    print("Analyzing plane quality...")
    quality_metrics = analyze_plane_quality(positions, plane_equation, normal, center)
    
    # Print results
    print("\n" + "=" * 60)
    print("PLANE OF BEST FIT RESULTS")
    print("=" * 60)
    
    a, b, c, d = plane_equation
    print(f"Plane Equation: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
    print(f"Normal Vector: [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
    print(f"Center Point: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
    print()
    print("Quality Metrics:")
    print(f"  Mean distance to plane: {quality_metrics['mean_distance']:.6f} meters")
    print(f"  Standard deviation: {quality_metrics['std_distance']:.6f} meters")
    print(f"  Maximum distance: {quality_metrics['max_distance']:.6f} meters")
    print(f"  Minimum distance: {quality_metrics['min_distance']:.6f} meters")
    print(f"  R-squared: {quality_metrics['r_squared']:.6f}")
    print("=" * 60)
    
    # Save results
    save_plane_results(plane_equation, normal, center, quality_metrics, positions)
    
    # Create visualization
    print("\nCreating visualization...")
    visualize_plane_and_points(positions, normal, center, plane_equation, quality_metrics)
    
    print("\nAnalysis complete! Check calib-results/plane_analysis.txt for detailed results.")
    print("Visualization saved as calib-results/plane_analysis.png") 