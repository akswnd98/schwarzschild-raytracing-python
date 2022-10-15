from time import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import cv2
from tqdm import tqdm
from numba import jit, njit, float64
from numba.np.extensions import cross2d

eps = 1e-5

pixel_width = 2560
pixel_height = 1440
aspect_ratio = pixel_width / pixel_height
theta = np.pi / 180. * 5.5
G = 6.673e-11
c = 299792458.
M = 1.98892e30 * 1e3
Rs = 2. * G * M / (c ** 2)
rin = 3. * Rs
rout = 15. * Rs
view_angle = 10. * np.pi / 180.
distance = 120. * Rs
r_bound=6400000000.

@jit
def get_eye_unit_vector (x, y, view_angle):
  ret = np.array([x - pixel_width / 2., y - pixel_height / 2., -1. / np.tan(view_angle / 2.) * (np.minimum(pixel_width, pixel_height) / 2.)], dtype=np.float64)
  return ret / np.linalg.norm(ret)

@njit
def get_rotate_x_mat (theta):
  return np.array([
    [1., 0., 0.],
    [0., np.cos(theta), -np.sin(theta)],
    [0., np.sin(theta), np.cos(theta)],
  ], dtype=np.float64)

@njit
def get_rotate_z_mat_by_pixel (x, y):
  if x == 0 and y == 0:
    return np.array([
      [1., 0., 0.],
      [0., 1., 0.],
      [0., 0., 1.],
    ], dtype=np.float64)
  xy_unit_vec = np.array([x, y], dtype=np.float64) / np.linalg.norm(np.array([x, y], dtype=np.float64))
  return np.array([
    [xy_unit_vec[1], xy_unit_vec[0], 0.],
    [-xy_unit_vec[0], xy_unit_vec[1], 0.],
    [0., 0., 1.],
  ], dtype=np.float64)

@njit
def get_accretion_plane (theta, pixel_x, pixel_y):
  a = np.array([0, 1, 0], dtype=np.float64)
  b = get_rotate_x_mat(theta)
  ret = np.dot(a, b)
  ret = np.dot(ret, get_rotate_z_mat_by_pixel(pixel_x, pixel_y))
  # ret = np.dot(np.dot(np.array([0, 1, 0], dtype=np.float64), get_rotate_x_mat(theta)), get_rotate_z_mat_by_pixel(pixel_x, pixel_y))
  ret = ret / np.linalg.norm(ret)
  return ret

@njit
def get_accretion_direction (theta, pixel_x, pixel_y):
  n = get_accretion_plane(theta, pixel_x, pixel_y)
  ret = np.array([-n[1], n[2]], dtype=np.float64)
  return ret

@njit
def f (_, x):
  return (x[1], -x[0] + 3. * G * M * (x[0] ** 2) / (c ** 2))

def check_outof_bound (t, y):
  return y[0] - 1. / r_bound
check_outof_bound.terminal = True

def check_inof_bound (t, y):
  return y[0] - 1. / (2. * G * M / (c ** 2.))
check_inof_bound.terminal = True

@njit
def get_uode0 (r0, phi0, ux0, uy0):
  r_p0 = ux0 * np.cos(phi0) + uy0 * np.sin(phi0)
  phi_p0 = (-ux0 * np.sin(phi0) + uy0 * np.cos(phi0)) / r0

  dr_dphi0 = (1. / phi_p0) * r_p0

  u0 = 1. / r0
  up0 = -dr_dphi0 / (r0 ** 2.)

  return np.array([u0, up0])

@njit
def ready_ode_t_eval (phi0, dense=0.001):
  return np.arange(phi0, phi0 + 4. * np.pi, dense)

def get_geodesic (r0, phi0, ux0, uy0, dense=0.001):
  uode0 = get_uode0(r0, phi0, ux0, uy0)
  t_eval = ready_ode_t_eval(phi0, dense)
  sol = solve_ivp(f, (phi0, phi0 + 4. * np.pi), uode0, t_eval=t_eval, dense_output=True, events=[check_outof_bound, check_inof_bound])

  return sol.t, sol.y[0]

@njit
def check_cross (vec, p1, p2):
  return cross2d(vec, p1) * cross2d(vec, p2) < 0.

@njit
def get_eye_xy (pixel_x, pixel_y, view_angle):
  eye = get_eye_unit_vector(pixel_x, pixel_y, view_angle)
  return np.array([eye[2], np.linalg.norm(np.array([eye[0], eye[1]], dtype=np.float64))], dtype=np.float64)

@njit
def check_accretion_direction_cross (accretion_direction, geodesic, rin, rout):
  for i in range(geodesic[0].shape[0] - 1):
    r = 1. / geodesic[1][i]
    p0 = np.array([r * np.cos(geodesic[0][i]), r * np.sin(geodesic[0][i])], dtype=np.float64)
    p1 = np.array([r * np.cos(geodesic[0][i + 1]), r * np.sin(geodesic[0][i + 1])], dtype=np.float64)
    if check_cross(accretion_direction, p0, p1) and r >= rin and r <= rout:
      return True
  
  return False

# @jit
# def check_accretion_direction_cross (accretion_direction, rin, rout, distance, pixel_x, pixel_y, view_angle):
#   eye = get_eye_unit_vector(pixel_x, pixel_y, view_angle)
#   eye_xy = np.array([eye[2], np.linalg.norm(np.array([eye[0], eye[1]], dtype=np.float64))], dtype=np.float64)
#   geodesic = get_geodesic(distance, 0., eye_xy[0], eye_xy[1])
#   for i in range(geodesic[0].shape[0] - 1):
#     r = 1. / geodesic[1][0][i]
#     p0 = np.array([r * np.cos(geodesic[0][i]), r * np.sin(geodesic[0][i])], dtype=np.float64)
#     p1 = np.array([r * np.cos(geodesic[0][i + 1]), r * np.sin(geodesic[0][i + 1])], dtype=np.float64)
#     if check_cross(accretion_direction, p0, p1) and r >= rin and r <= rout:
#       return True
  
#   return False

start_line = 0

def execute_drawing_pipeline (start_line):
  img = np.zeros([pixel_height, pixel_width], dtype=np.uint8) if start_line == 0 else cv2.imread('line{}.png'.format(start_line), cv2.IMREAD_GRAYSCALE)
  for y in tqdm(range(start_line, img.shape[0])):
    for x in range(img.shape[1]):
      eye_xy = get_eye_xy(x, y, view_angle)
      if x == img.shape[1] // 2 and y == img.shape[0] // 2:
        img[y][x] = 0
        continue
      geodesic = get_geodesic(distance, 0., eye_xy[0], eye_xy[1])
      if check_accretion_direction_cross(get_accretion_direction(theta, x - img.shape[1] / 2., y - img.shape[0] / 2.), np.array([geodesic[0], geodesic[1]]), rin, rout):
        img[y][x] = 255
      else:
        img[y][x] = 0
    
    if y % 10 == 10 - 1:
      cv2.imwrite('line{}.png'.format(y + 1), img)

  cv2.imwrite('final.png', img)

execute_drawing_pipeline(start_line)
