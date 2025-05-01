import open3d as o3d
import numpy as np
import sys

# --- PARAMETERS ---
PLANE_THRESH = 0.001              # Increased from 0.005 for LEGO blocks
SAMPLE_POINTS = 200000           
CLUSTER_THRESH = 0.005           
MIN_CLUSTER_SIZE = 50           # Increased minimum cluster size
OBJECT_HEIGHT_THRESH = 0.001      # Minimum height above plane to be considered object
# -------------------

def remove_floor_completely(mesh):
    """Enhanced floor removal for LEGO blocks"""
    # 1. Sample points and detect plane
    pcd = mesh.sample_points_uniformly(number_of_points=SAMPLE_POINTS)
    pcd.estimate_normals()
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=PLANE_THRESH,
        ransac_n=3,
        num_iterations=1000
    )
    a, b, c, d = plane_model
    print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    # 2. Compute distances for all vertices
    vertices = np.asarray(mesh.vertices)
    distances = (a*vertices[:,0] + b*vertices[:,1] + c*vertices[:,2] + d) / np.sqrt(a**2 + b**2 + c**2)
    
    # 3. Get initial object mask using height threshold
    object_vert_mask = distances > OBJECT_HEIGHT_THRESH
    
    # 4. If no points remain, try with smaller threshold
    if not np.any(object_vert_mask):
        print("Warning: No points above threshold, trying with reduced threshold")
        object_vert_mask = distances > (OBJECT_HEIGHT_THRESH / 2)
    
    # 5. Convert to point cloud for cluster analysis
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(vertices[object_vert_mask])
    
    # 6. Cluster analysis to remove floating floor fragments
    labels = np.array(object_pcd.cluster_dbscan(
        eps=CLUSTER_THRESH, 
        min_points=MIN_CLUSTER_SIZE, 
        print_progress=True))
    
    if len(labels) == 0:
        raise RuntimeError("No clusters found - adjust parameters")
    
    # 7. Find all significant clusters (not just the largest)
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        raise RuntimeError("No valid clusters found")
    
    # Get clusters with significant size
    significant_clusters = unique_labels[counts > (0.1 * len(labels))]
    
    # 8. Create refined object mask
    cluster_mask = np.isin(labels, significant_clusters)
    refined_object_mask = np.zeros(len(vertices), dtype=bool)
    refined_object_mask[np.where(object_vert_mask)[0][cluster_mask]] = True
    
    # 9. Create final mesh
    triangles_np = np.asarray(mesh.triangles)
    final_tri_mask = np.all(refined_object_mask[triangles_np], axis=1)
    
    final_mesh = o3d.geometry.TriangleMesh()
    final_mesh.vertices = mesh.vertices
    final_mesh.triangles = o3d.utility.Vector3iVector(triangles_np[final_tri_mask])
    
    if mesh.has_vertex_colors():
        final_mesh.vertex_colors = mesh.vertex_colors
    if mesh.has_vertex_normals():
        final_mesh.vertex_normals = mesh.vertex_normals
    
    final_mesh.remove_unreferenced_vertices()
    return final_mesh, plane_model

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 script.py input_mesh.ply output_mesh.ply")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(input_filename)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh: {input_filename}")
    mesh.compute_vertex_normals()

    # Process mesh
    object_mesh, plane_model = remove_floor_completely(mesh)
    
    # Save results
    o3d.io.write_triangle_mesh(output_filename, object_mesh)
    print(f"Saved clean object mesh to: {output_filename}")

    # Visualize
    o3d.visualization.draw_geometries([object_mesh],
        window_name="Cleaned LEGO Object",
        width=800, height=600
    )

if __name__ == "__main__":
    main() 
