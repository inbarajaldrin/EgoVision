# #!/usr/bin/env python3
import open3d as o3d
import sys

def view_ply(filepath):
    print(f"Loading {filepath}...")
    pcd = o3d.io.read_point_cloud(filepath)
    
    if not pcd.points:
        print("Error: Empty point cloud or file not found!")
        return
    
    print(f"Point count: {len(pcd.points)}")
    print(f"Has colors: {pcd.has_colors()}")
    print("\nControls: Rotate (Left drag) | Zoom (Mouse wheel) | Pan (Right drag) | Exit (Q/ESC)")
    
    o3d.visualization.draw_geometries([pcd], window_name=filepath)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 view_ply.py your_file.ply")
        sys.exit(1)
    view_ply(sys.argv[1])

