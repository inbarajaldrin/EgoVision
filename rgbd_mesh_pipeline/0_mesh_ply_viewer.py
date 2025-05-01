import open3d as o3d
import sys

def view_mesh(filename):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(filename)
    if mesh.is_empty():
        print(f"Failed to load mesh: {filename}")
        sys.exit(1)
    
    # Optionally compute normals (for better lighting if they don't exist)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    # View the mesh
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Mesh Viewer",
        width=800,
        height=600,
        mesh_show_back_face=True  # Optional: shows back faces too
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 view_mesh.py input_mesh.ply")
        sys.exit(1)
    
    input_mesh = sys.argv[1]
    view_mesh(input_mesh)
