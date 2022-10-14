from time import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np

G = 6.673e-11
M = 1.98892e30 * 1e5
c = 299792458

def f (_, x):
  return (x[1], -x[0] + 3 * G * M * (x[0] ** 2) / (c ** 2))

def e (_, x):
  return 1 - (2 * G * M) / ((1 / x[0]) * (c ** 2))

def get_geodesic (M, r0, phi0, ux0, uy0, dense=0.001, r_bound=8000000000):
  def f (_, x):
    return (x[1], -x[0] + 3 * G * M * (x[0] ** 2) / (c ** 2))

  r_p0 = ux0 * np.cos(phi0) + uy0 * np.sin(phi0)
  phi_p0 = (-ux0 * np.sin(phi0) + uy0 * np.cos(phi0)) / r0

  dr_dphi0 = 1 / phi_p0 * r_p0

  u0 = 1 / r0
  up0 = -dr_dphi0 / r0 ** 2

  u_ode0 = [u0, up0]

  def check_outof_bound (t, y):
    return 1 / y[0] - r_bound
  check_outof_bound.terminal = True
  
  def check_inof_bound (t, y):
    return 1 / y[0] - 2 * G * M / (c ** 2)
  check_inof_bound.terminal = True

  t_eval = np.arange(phi0, phi0 + 4 * np.pi, dense)
  start = time()
  sol = solve_ivp(f, (phi0, phi0 + 4 * np.pi), u_ode0, t_eval=t_eval, dense_output=True, events=[check_outof_bound, check_inof_bound])
  end = time()

  print(end - start)

  return sol.t, sol.y

plt.axis('equal')
def plot_black_hole_radius (M):
  G = 6.673e-11
  c = 299792458
  r = 2 * G * M / (c ** 2)
  theta = np.arange(0, 2 * np.pi, 0.01)
  x = r * np.cos(theta)
  y = r * np.sin(theta)
  plt.plot(x, y)
plot_black_hole_radius(M)

geodesic = get_geodesic(M, 800000000, 0 * np.pi / 180, -1, 1)
r = 1 / geodesic[1][0]
plt.plot(r * np.cos(geodesic[0]), r * np.sin(geodesic[0]))
plt.show()

# geodesic = get_geodesic(M, 800000000, 30 * np.pi / 180, -1, 0.47, atol=1000000000000000000000000, rtol=10000000000000000000000000000000)
# r = 1 / geodesic[1][:, 0]
# plt.plot(r * np.cos(geodesic[0]), r * np.sin(geodesic[0]))

# plt.show()
