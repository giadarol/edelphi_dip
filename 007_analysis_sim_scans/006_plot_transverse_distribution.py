import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import e as qe

import PyECLOUD.myfilemanager as mfm

import PyECLOUD.geom_impact_poly_fast_impact as geom
import PyPIC.FiniteDifferences_ShortleyWeller_SquareGrid as FDSW

chamber = geom.polyg_cham_geom_object(
        filename_chm = '../reference_simulation/pyecloud_config/LHC_chm_ver.mat',
        flag_non_unif_sey = False)

pic = FDSW.FiniteDifferences_ShortleyWeller_SquareGrid(chamb=chamber, Dh=0.1e-3)

obp = mfm.myloadmat_to_obj('../quad_combined_distribution/combined_distribution_sey_1.40_VRF_4MV_intensity_1.2e11ppb_450GeV_N_mp_500000_symm.mat')

pic.scatter(x_mp=obp.x_mp, y_mp=obp.y_mp, nel_mp=obp.nel_mp)

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.122, top=.82)
mbl = ax.pcolormesh(1e3*pic.xg, 1e3*pic.yg, -pic.rho.T/qe, cmap=plt.cm.jet, vmax=5e13)
ax.plot(1e3*chamber.Vx, 1e3*chamber.Vy, lw=2, color='yellow')
cb = plt.colorbar(mbl, ax=ax)
plt.axis('equal')
ax.set_xlim(1e3*np.min(pic.xg), 1e3*np.max(pic.xg))
ax.set_ylim(1e3*np.min(pic.yg), 1e3*np.max(pic.yg))
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
cb.ax.set_ylabel(r'Charge density [e$^{-}$/m$^3$]')

plt.show()
