from time import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import cv2
from tqdm import tqdm

eps = 1e-5

pixel_width = 320
pixel_height = 180
aspect_ratio = pixel_width / pixel_height
theta = np.pi / 180 * 5.5
G = 6.673e-11
c = 299792458
M = 1.98892e30 * 1e3
Rs = 2 * G * M / (c ** 2)
rin = 3 * Rs
rout = 15 * Rs
view_angle = 12 * np.pi / 180
distance = 120 * Rs

def get_eye_unit_vector (x, y, view_angle):
  ret = np.array([x - pixel_width / 2, y - pixel_height / 2, -1 / np.tan(view_angle / 2) * (np.minimum(pixel_width, pixel_height) / 2)], dtype=np.float64)
  return ret / np.linalg.norm(ret)

def get_rotate_x_mat (theta):
  return np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)],
  ], dtype=np.float64)

def get_rotate_z_mat_by_pixel (x, y):
  if x == 0 and y == 0:
    return np.array([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ], dtype=np.float64)
  xy_unit_vec = np.array([x, y], dtype=np.float64) / np.linalg.norm(np.array([x, y], dtype=np.float64))
  return np.array([
    [xy_unit_vec[1], xy_unit_vec[0], 0],
    [-xy_unit_vec[0], xy_unit_vec[1], 0],
    [0, 0, 1],
  ], dtype=np.float64)

def get_accretion_plane (theta, pixel_x, pixel_y):
  ret = (
    np.array([0, 1, 0], dtype=np.float64)
    @ get_rotate_x_mat(theta)
    @ get_rotate_z_mat_by_pixel(pixel_x, pixel_y)
  )
  ret = ret / np.linalg.norm(ret)
  return ret

def get_accretion_direction (theta, pixel_x, pixel_y):
  n = get_accretion_plane(theta, pixel_x, pixel_y)
  ret = np.array([-n[1], n[2]], dtype=np.float64)
  return ret

def get_geodesic (M, r0, phi0, ux0, uy0, dense=0.01, r_bound=6400000000):
  def f (_, x):
    return (x[1], -x[0] + 3 * G * M * (x[0] ** 2) / (c ** 2))

  r_p0 = ux0 * np.cos(phi0) + uy0 * np.sin(phi0)
  phi_p0 = (-ux0 * np.sin(phi0) + uy0 * np.cos(phi0)) / r0

  dr_dphi0 = (1 / phi_p0) * r_p0

  u0 = 1 / r0
  up0 = -dr_dphi0 / (r0 ** 2)

  u_ode0 = [u0, up0]

  def check_outof_bound (t, y):
    return 1 / y[0] - r_bound
  check_outof_bound.terminal = True
  
  def check_inof_bound (t, y):
    return 1 / y[0] - 2 * G * M / (c ** 2)
  check_inof_bound.terminal = True

  t_eval = np.arange(phi0, phi0 + 4 * np.pi, dense)
  sol = solve_ivp(f, (phi0, phi0 + 4 * np.pi), u_ode0, t_eval=t_eval, dense_output=True, events=[check_outof_bound, check_inof_bound])

  return sol.t, sol.y

def check_cross (vec, p1, p2):
  return np.cross(vec, p1) * np.cross(vec, p2) < 0

def check_accretion_direction_cross (M, accretion_direction, rin, rout, distance, pixel_x, pixel_y, view_angle):
  eye = get_eye_unit_vector(pixel_x, pixel_y, view_angle)
  eye_xy = np.array([eye[2], np.linalg.norm([eye[0], eye[1]])], dtype=np.float64)
  geodesic = get_geodesic(M, distance, 0, eye_xy[0], eye_xy[1], r_bound=distance * 10)
  for i in range(geodesic[0].shape[0] - 1):
    r = 1 / geodesic[1][0][i]
    p0 = [r * np.cos(geodesic[0][i]), r * np.sin(geodesic[0][i])]
    p1 = [r * np.cos(geodesic[0][i + 1]), r * np.sin(geodesic[0][i + 1])]
    if check_cross(accretion_direction, p0, p1) and r >= rin and r <= rout:
      return True
  
  return False


# for theta in np.arange(30, 90, 6):
#   img = np.zeros([pixel_height, pixel_width], dtype=np.uint8)
#   for y in tqdm(range(img.shape[0])):
#     if y == 90:
#       continue
#     for x in range(img.shape[1]):
#       if check_accretion_direction_cross(M, get_accretion_direction(theta, x - img.shape[1] / 2, y - img.shape[0] / 2), rin, rout, distance, x, y, view_angle):
#         img[y][x] = 255
#       else:
#         img[y][x] = 0

#   cv2.imwrite('accretion-plane-theta{}.png'.format(theta), img)

img = np.zeros([pixel_height, pixel_width], dtype=np.uint8)
for y in tqdm(range(img.shape[0])):
  if y == 90:
    continue
  for x in range(img.shape[1]):
    if check_accretion_direction_cross(M, get_accretion_direction(theta, x - img.shape[1] / 2, y - img.shape[0] / 2), rin, rout, distance, x, y, view_angle):
      img[y][x] = 255
    else:
      img[y][x] = 0

cv2.imwrite('accretion-plane.png'.format(theta), img)

# for m, d in zip([
#   1.98892e30 * 1e3, 1.98892e30 * 1e4, 1.98892e30 * 1e5, 1.98892e30 * 1e6,
#   1.98892e30 * 1e7, 1.98892e30 * 1e8, 1.98892e30 * 1e9, 1.98892e30 * 1e10
# ], [
#   1000000000, 10000000000, 100000000000, 1000000000000, 10000000000000,
#   100000000000000, 1000000000000000, 10000000000000000
# ]):
#   img = np.zeros([pixel_height, pixel_width], dtype=np.uint8)
#   for y in tqdm(range(img.shape[0])):
#     if y == 90:
#       continue
#     for x in range(img.shape[1]):
#       if check_accretion_direction_cross(m, get_accretion_direction(theta, x - img.shape[1] / 2, y - img.shape[0] / 2), rin, rout, d, x, y, view_angle):
#         img[y][x] = 255
#       else:
#         img[y][x] = 0

#   cv2.imwrite('accretion-plane-m:{}-d:{}.png'.format(m, d), img)

# geodesic = get_geodesic(M, 640000000, 0, -1, 0.04)
# plt.axis('equal')
# r = 1 / geodesic[1][0]
# plt.plot(r * np.cos(geodesic[0]), r * np.sin(geodesic[0]))

# def plot_black_hole_radius (M):
#   G = 6.673e-11
#   c = 299792458
#   r = 2 * G * M / (c ** 2)
#   theta = np.arange(0, 2 * np.pi, 0.01)
#   x = r * np.cos(theta)
#   y = r * np.sin(theta)
#   plt.plot(x, y)

# for i in np.arange(0.01, 1, 0.01):
#   geodesic = get_geodesic(M, 6400000000, 0, -1, i)
#   plt.axis('equal')
#   r = 1 / geodesic[1][0]
#   plt.plot(r * np.cos(geodesic[0]), r * np.sin(geodesic[0]))

# plt.plot([rout * np.cos(-theta), rout * np.cos(-theta + np.pi)], [rout * np.sin(-theta), rout * np.sin(-theta + np.pi)])

# plot_black_hole_radius(M)
# plt.show()
