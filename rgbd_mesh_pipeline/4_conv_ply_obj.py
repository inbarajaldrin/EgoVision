import pymeshlab
import sys

if len(sys.argv) != 3:
    print("Usage: python3 script.py input_mesh.ply output_mesh.obj")
    sys.exit(1)

input_ply = sys.argv[1]
output_obj = sys.argv[2]

# Create a new MeshSet
ms = pymeshlab.MeshSet()

# Load your mesh
ms.load_new_mesh(input_ply)

# Apply UV parametrization
ms.apply_filter('compute_texcoord_parametrization_triangle_trivial_per_wedge')

# Transfer vertex colors to texture
texture_name = output_obj.replace('.obj', '.png')
ms.apply_filter('transfer_attributes_to_texture_per_vertex',
                textname=texture_name, textw=1024, texth=1024)

# Save the mesh with texture coordinates
ms.save_current_mesh(output_obj,
                     save_vertex_color=False,
                     save_wedge_texcoord=True)