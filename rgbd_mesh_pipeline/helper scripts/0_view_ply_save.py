import open3d as o3d
import sys

def view_and_save_pcd(filepath, save_path="saved_pcd_view.png"):
    print(f"Loading {filepath}...")
    pcd = o3d.io.read_point_cloud(filepath)

    if not pcd.has_points():
        print("Error: Empty point cloud or file not found!")
        return

    print(f"Point count: {len(pcd.points)}")
    print(f"Has colors: {pcd.has_colors()}")
    print("Controls: Rotate (Left drag) | Zoom (Scroll) | Pan (Right drag) | Save (S) | Exit (ESC/Q)")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud Viewer", width=1024, height=768)
    vis.add_geometry(pcd)

    def save_screen(vis):
        vis.capture_screen_image(save_path)
        print(f"✅ Screenshot saved to: {save_path}")
        return False

    vis.register_key_callback(ord("S"), save_screen)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 0_view_ply_save.py your_pointcloud.ply [output_image.png]")
        sys.exit(1)

    filepath = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) >= 3 else "saved_pcd_view.png"
    view_and_save_pcd(filepath, save_path)
