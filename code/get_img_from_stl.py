import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import copy
np.random.seed(42)

def capture_random_views(mesh, n, save_dir, width=224, height=224):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="STL", width=width, height=height, visible=True)
    mesh.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
    vis.add_geometry(mesh)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])
    image_count = 0
    running = True
    for group in range(3):
        for angle in range(0, 360, 15):
            if group == 0:
                R = mesh.get_rotation_matrix_from_xyz((np.radians(angle), 0, 0))
            elif group == 1:
                R = mesh.get_rotation_matrix_from_xyz((0, np.radians(angle), 0))
            else:
                R = mesh.get_rotation_matrix_from_xyz((0, 0, np.radians(angle)))
            mesh_rotated = copy.deepcopy(mesh)
            mesh_rotated.rotate(R, center=mesh_rotated.get_center())
            vis.clear_geometries()
            vis.add_geometry(mesh_rotated)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)
            image = vis.capture_screen_float_buffer(False)
            img = (np.asarray(image) * 255).astype(np.uint8)
            img = Image.fromarray(img)
            img = img.convert('L')
            img.save(os.path.join(save_dir, f"{group}_{image_count}.png"))
            image_count+=1
        image_count = 0

    vis.destroy_window()

if __name__ == "__main__":
    classes = ['01_8','03_3','04_7', '05_5', '06_10', '07_4', '08_9', '09_6', '10_1']
    num_images = 12
    root_output_path = f"/media/chikazoe/HDPH-UT/HACHIX/backups/images-stl/colorless"
    for class_name in classes:
        folder_name = class_name.split('_')[0]
        output_path = os.path.join(root_output_path, folder_name)
        os.makedirs(output_path, exist_ok=True)
        stl_path = f"/media/chikazoe/HDPH-UT/HACHIX/backups/01_8-20240822T004733Z-001/01_8/{class_name}.stl"
        mesh = o3d.io.read_triangle_mesh(stl_path)
        mesh = mesh.compute_vertex_normals()
        capture_random_views(mesh, num_images, output_path, 1280, 960)

