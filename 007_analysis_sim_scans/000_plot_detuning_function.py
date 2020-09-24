import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as clight

from scipy.constants import e as qe

import PyECLOUD.myfilemanager as mfm
import PyECLOUD.mystyle as ms

n_phi = 360
n_r = 200
beta_fun = 92.7
omega0 = 2*np.pi*clight/27e3 # Revolution angular frquency
omega_s = 4.9e-3*omega0
sigma_b = 0.097057
vmax_edens = 1.5e14

ob = mfm.myloadmat_to_obj('../000a_sin_response_unperturbed_pinch/linear_strength_y.mat')
z_slices = ob.z_slices
p = np.polyfit(ob.z_slices, ob.k_z_integrated, deg=10)

alpha_N = p[::-1] # Here I fit the strength
A_N = -beta_fun * alpha_N/4/ np.pi # I go to the coefficient in the tune
N_terms = len(A_N)
r_max = np.max(np.abs(z_slices))
dz = z_slices[1] - z_slices[0]

r_vect = np.linspace(0, r_max, n_r)
phi_vect = np.linspace(0, 2*np.pi, n_phi+1)[:-1]
dphi = phi_vect[1] - phi_vect[0]

sin_PHI = np.sin(phi_vect)
cos_PHI = np.cos(phi_vect)

C_N_PHI = np.zeros((N_terms, n_phi))

for nn in range(N_terms):
    if nn == 0:
        C_N_PHI[nn, :] = phi_vect
        continue
    if nn == 1:
        C_N_PHI[nn, :] = sin_PHI
        continue
    C_N_PHI[nn, :] = cos_PHI**(nn-1)*sin_PHI/nn + (nn-1)/nn * C_N_PHI[nn-2, :]

dPhi_R_PHI = np.zeros((n_r, n_phi))

for nn in range(N_terms):
    dPhi_R_PHI += -omega0/omega_s * A_N[nn] * np.dot(
            np.atleast_2d(r_vect**nn).T, np.atleast_2d(C_N_PHI[nn, :]))


d_Q_R_PHI = -omega_s/omega0 * np.diff(dPhi_R_PHI[:, :], axis=1)/dphi

deltascaled_obs = 3e-2
r_obs = np.sqrt(deltascaled_obs**2 + z_slices**2)
phi_obs = np.arctan2(deltascaled_obs, z_slices)
from scipy.interpolate import interp2d
dQ_obs_fun = interp2d(r_vect, phi_vect[:-1], d_Q_R_PHI.T)
dQ_obs = np.squeeze(np.array([ dQ_obs_fun(rr, pp) for rr,pp in zip(r_obs, phi_obs)]))
k_obs = - dQ_obs*4*np.pi/beta_fun

plt.close('all')
ms.mystyle_arial(fontsz=14, dist_tick_lab=5, traditional_look=False)
fig100 = plt.figure(100, figsize=(6.4, 4.8*1.5))
ax100 = plt.subplot2grid(shape=(2,1), loc=(0,0),
        rowspan=1, colspan=1, fig=fig100)
ax101 = fig100.add_subplot(212, sharex=ax100)
ax101.plot(100*ob.z_slices, -1/(4*np.pi)*beta_fun*ob.k_z_integrated,
        lw=3, color='C0', alpha=.6, label='Simulation')
ax101.plot(100*ob.z_slices, dQ_obs, linestyle='--',
        lw=3, color='C1', alpha=.6, label=r'Polynomial fit (N$_P$=10)')

ax101.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
ax101.set_xlim(-30, 30)
ax101.set_ylim(bottom=0)
ax101.set_xlabel('z [cm]')
ax101.set_ylabel('Tune deviation ($\Delta Q$)')
#ax101.grid(True, linestyle=':')
ax101.legend(loc='lower center', ncol=1, fontsize='medium', frameon=False)
#ax101.legend(loc='lower left', ncol=2, fontsize='medium', frameon=False,
#        bbox_to_anchor=(0,-0.04))
fig100.subplots_adjust(bottom=.12)

obmap = mfm.myloadmat_to_obj('../003_generate_field_map/rho_map_ec0.mat')
ix_zero = np.argmin(np.abs(obmap.xg))

mpbl = ax100.pcolormesh(1e2*obmap.zg, 1e3*obmap.yg,
        -(1e-14/qe)*np.mean(obmap.rho[:,ix_zero-1: ix_zero+2, :], axis=1).T,
        )
        #vmin=0, vmax=vmax_edens*1e-14)
ax100.set_ylim(-4.5, 4.5)
ax100.set_ylabel('x [mm]')

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax100)
cax = divider.new_vertical(size="10%", pad=0.1, pack_start=False)
fig100.add_axes(cax)
cbar = fig100.colorbar(mpbl, cax=cax, orientation="horizontal")
cax.xaxis.set_ticks_position('top')
cbar.ax.set_xlabel(r'Charge density [10$^{14}$ e$^-$/m$^{3}$]', labelpad=8)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_tick_params(pad=0.5)
fig100.subplots_adjust(bottom=.12, hspace=.25)
plt.show()


lambda_b = np.exp(-z_slices**2/(2*sigma_b**2))
wavg_DQ = np.sum(lambda_b*dQ_obs)/np.sum(lambda_b)
