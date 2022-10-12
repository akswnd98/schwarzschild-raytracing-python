import numpy as np

eps = 1e-5

pixel_width = 1920
pixel_height = 1080
aspect_ratio = pixel_width / pixel_height
theta = np.pi / 180 * 30

def get_camera_ray_unit_vector (x, y):
  ret = np.array([x - pixel_width // 2, y - pixel_height // 2, -np.minimum(pixel_width, pixel_height) // 2], dtype=np.float64)
  return ret / np.linalg.norm(ret)

def get_rotate_x_mat (theta):
  return np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)],
  ], dtype=np.float64)

def get_rotate_z_mat_by_pixel (x, y):
  xy_unit_vec = np.linalg.norm(np.array([x, y], dtype=np.float64)) / np.array([x, y], dtype=np.float64)
  return np.array([
    [xy_unit_vec[0], xy_unit_vec[1], 0],
    [-xy_unit_vec[1], xy_unit_vec[0], 0],
    [0, 0, 1],
  ], dtype=np.float64)

def get_accretion_plane_norm_vector (theta, pixel_x, pixel_y):
  return np.linalg.norm(
    np.array([0, 1, 0], dtype=np.float64)
    @ get_rotate_x_mat(theta)
    @ get_rotate_z_mat_by_pixel(pixel_x, pixel_y)
  )

def get_accretion_direction_norm_vector (theta, pixel_x, pixel_y):
  n = get_accretion_plane_norm_vector(theta, pixel_x, pixel_y)
  ret = np.array([n[3], -n[2]], dtype=np.float64)
  ret = ret / np.linalg.norm(ret)
  return ret

def get_path ()
def check_is_crossing_accretion_direction (theta, pixel_x, pixel_y):
  
