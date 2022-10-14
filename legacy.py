def get_geodesic (M, r0, phi0, ux0, uy0, error_check_dense=0.02, dense=0.01):
  def f (_, x):
    return (x[1], -x[0] + 3 * G * M * (x[0] ** 2) / (c ** 2))

  r_p0 = ux0 * np.cos(phi0) + uy0 * np.sin(phi0)
  phi_p0 = (-ux0 * np.sin(phi0) + uy0 * np.cos(phi0)) / r0

  dr_dphi0 = 1 / phi_p0 * r_p0

  u0 = 1 / r0
  up0 = -dr_dphi0 / r0 ** 2

  phi_ode = phi0
  u_ode0 = [u0, up0]

  phi_step = np.arange(phi0, phi0 + np.pi * 8, error_check_dense, dtype=np.float64)

  phi_ode = [phi0]
  u_ode = [np.array(u_ode0, dtype=np.float64)]

  for i in range(len(phi_step) - 1):
    t_eval=np.arange(phi_step[i], phi_step[i + 1], dense, dtype=np.float64)
    t_eval[0] = phi_step[i]
    t_eval[-1] = phi_step[i + 1]
    sol = solve_ivp(f, (phi_step[i], phi_step[i + 1]), u_ode[-1], dense_output=True, t_eval=t_eval)
    if 1 / sol.y[0, -1] < 2 * G * M / (c ** 2) or not sol.success:
      break
    phi_ode += list(sol.t)
    u_ode += list(np.transpose(sol.y))

  phi_ode = np.array(phi_ode)
  u_ode = np.array(u_ode)

  u = u_ode[:, 0]
  r = 1 / u

  return phi_ode, u_ode