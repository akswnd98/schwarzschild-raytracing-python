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

y0 = 800000000
phi0 = 30 * np.pi / 180
r0 = y0 / np.sin(phi0)
rp0 = -y0 / (np.tan(phi0) * np.sin(phi0))
u0 = 1 / r0
up0 = -rp0 / (r0 ** 2)

phi_ode = phi0
u_ode0 = [u0, up0]

phi_step = np.arange(phi0, phi0 + np.pi * 8, 0.1, dtype=np.float64)

phi_ode = [phi0]
u_ode = [np.array(u_ode0, dtype=np.float64)]

for i in range(1, len(phi_step)):
  sol = solve_ivp(f, (phi_step[i - 1], phi_step[i]), u_ode[-1], dense_output=True)
  if 1 / sol.y[0, -1] < 2 * G * M / (c ** 2) or not sol.success:
    break
  phi_ode += list(sol.t)
  u_ode += list(np.transpose(sol.y))

phi_ode = np.array(phi_ode)
u_ode = np.array(u_ode)

u = u_ode[:, 0]
r = 1 / u

plt.axis('equal')
plt.plot(r * np.cos(phi_ode), r * np.sin(phi_ode))
plt.show()

print(u_ode)

G = 6.673e-11
c = 299792458
M = 1.98892e30 * 1e5
