#!/usr/bin/env python3
import sys
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering


def load_geometry(path):
    """
    Load a triangle mesh if it has faces; otherwise load as a point cloud.
    Returns (geometry, is_mesh_flag).
    """
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        pcd = o3d.io.read_point_cloud(path)
        return pcd, False
    return mesh, True


def main(mesh1_path, mesh2_path, out_path):
    # Initialize GUI
    app = gui.Application.instance
    app.initialize()

    # Create window
    window = app.create_window("Interactive Align", 1280, 800)

    # Create 3D scene widget
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene_widget)

    # Load geometries
    geom1, is_mesh1 = load_geometry(mesh1_path)
    geom2, is_mesh2 = load_geometry(mesh2_path)

    # Materials
    mat1 = rendering.MaterialRecord(); mat1.shader = "defaultLit"
    mat2 = rendering.MaterialRecord(); mat2.shader = "defaultLit"
    mat2.base_color = (1, 0, 0, 1)  # tint second geometry red

    # Add geometries to scene
    scene_widget.scene.add_geometry("geom1", geom1, mat1)
    scene_widget.scene.add_geometry("geom2", geom2, mat2)

    # Frame camera on first geometry
    bbox = geom1.get_axis_aligned_bounding_box()
    scene_widget.setup_camera(60, bbox, bbox.get_center())

    # Build UI control panel
    em = window.theme.font_size
    ctrl_panel = gui.Vert(em, gui.Margins(em, em, em, em))
    window.add_child(ctrl_panel)

    # Geometry selector
    combo = gui.Combobox()
    combo.add_item("geom1")
    combo.add_item("geom2")
    combo.selected_index = 1
    ctrl_panel.add_child(gui.Label("Select Geometry:"))
    ctrl_panel.add_child(combo)

    # Create sliders and text boxes for each parameter
    params = [
        ("Rotate X", -180, 180),
        ("Rotate Y", -180, 180),
        ("Rotate Z", -180, 180),
        ("Trans X", -0.5, 0.5),
        ("Trans Y", -0.5, 0.5),
        ("Trans Z", -0.5, 0.5),
    ]
    sliders = {}
    edits = {}
    for name, mn, mx in params:
        # Horizontal layout for label, slider, and number edit
        row = gui.Horiz(em)
        row.add_child(gui.Label(name))
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(mn, mx)
        slider.double_value = 0.0
        row.add_child(slider)
        num_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        num_edit.double_value = 0.0
        row.add_child(num_edit)
        ctrl_panel.add_child(row)
        sliders[name] = slider
        edits[name] = num_edit

        # Callback when slider moves
        def make_slider_cb(nm):
            def cb(val):
                edits[nm].double_value = sliders[nm].double_value
                update_transform()
            return cb
        slider.set_on_value_changed(make_slider_cb(name))

        # Callback when text changes (accept one arg)
        def make_edit_cb(nm, mn_val, mx_val):
            def cb(val):
                v = edits[nm].double_value
                v = max(min(v, mx_val), mn_val)
                sliders[nm].double_value = v
                update_transform()
            return cb
        num_edit.set_on_value_changed(make_edit_cb(name, mn, mx))

    # Update transform based on current controls
    def update_transform():
        rx = np.deg2rad(sliders["Rotate X"].double_value)
        ry = np.deg2rad(sliders["Rotate Y"].double_value)
        rz = np.deg2rad(sliders["Rotate Z"].double_value)
        tx = sliders["Trans X"].double_value
        ty = sliders["Trans Y"].double_value
        tz = sliders["Trans Z"].double_value
        # Rotation matrices
        Rx = o3d.geometry.get_rotation_matrix_from_xyz((rx, 0, 0))
        Ry = o3d.geometry.get_rotation_matrix_from_xyz((0, ry, 0))
        Rz = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, rz))
        R = Rz @ Ry @ Rx
        # Full transform
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = (tx, ty, tz)
        target = "geom1" if combo.selected_index == 0 else "geom2"
        scene_widget.scene.set_geometry_transform(target, T)

    # Merge & Save button
    def on_save():
        g1, m1 = load_geometry(mesh1_path)
        g2, m2 = load_geometry(mesh2_path)
        tf = scene_widget.scene.get_geometry_transform("geom2")
        g2.transform(tf)
        merged = g1 + g2
        if m1 and m2:
            o3d.io.write_triangle_mesh(out_path, merged)
        else:
            o3d.io.write_point_cloud(out_path, merged)
        print(f"[✔] Saved merged geometry to {out_path}")
    save_btn = gui.Button("Merge & Save")
    save_btn.set_on_clicked(on_save)
    ctrl_panel.add_child(save_btn)

    # Layout positions
    def on_layout(layout_context):
        r = window.content_rect
        panel_w = 350
        ctrl_panel.frame = gui.Rect(r.x, r.y, panel_w, r.height)
        scene_widget.frame = gui.Rect(r.x + panel_w, r.y,
                                      r.width - panel_w, r.height)
    window.set_on_layout(on_layout)

    app.run()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python manual_rotate.py mesh1.ply mesh2.ply output.ply")
        sys.exit(1)
    main(*sys.argv[1:])
