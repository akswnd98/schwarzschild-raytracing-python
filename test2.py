import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np

G = 6.673e-11
M = 1.98892e35 * 1.825465
c = 299792458
Rs = 2 * G * M / c ** 2

def mu_func (_, u):
  r = 1 / u[0]
  rp = -u[1] / (u[0] ** 2)
  phip_square = (1 - rp ** 2) / (r ** 2)
  return (
    u[1],
    ((-u[0] + 3 / 2 * Rs * u[0] ** 2) * phip_square ** 2 + 4 * rp * u[1] ** 3 / ((1 - rp ** 2) * u[0] ** 3) - 2 * rp / r) / (1 + 2 * rp * u[0] * u[1] / ((1 - rp ** 2) * u[0] ** 3))
  )

y0 = 1392700000
phi0 = np.pi / 10;
rp_phi0 = -y0 / (np.tan(phi0) * np.sin(phi0))
r0 = y0 / np.sin(phi0)
phip0 = np.sqrt(1 / (rp_phi0 ** 2 + r0 ** 2))
rp0 = -y0 / (np.tan(phi0) * np.sin(phi0)) * phip0
u0 = 1 / r0
up0 = -rp0 / (r0 ** 2)

u_ode0 = [u0, up0]

s_ode = [0]
u_ode = [u_ode0]

s_step = np.arange(0, r0 * 3, r0 * 3 / 200, dtype=np.float64)
for i in range(1, len(s_step)):
  sol = solve_ivp(mu_func, (s_step[i - 1], s_step[i]), u_ode[-1], dense_output=True)
  if 1 / sol.y[0, -1] < 2 * G * M / (c ** 2) or not sol.success:
    break
  s_ode += list(sol.t)
  u_ode += list(np.transpose(sol.y))

s_ode = np.array(s_ode)
u_ode = np.array(u_ode)

u = u_ode[:, 0]
r = 1 / u

plt.axis('equal')
plt.plot(r * np.cos(s_ode), r * np.sin(s_ode))
plt.show()
