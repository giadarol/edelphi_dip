import numpy as np
from PyHEADTAIL.general.element import Element

class ModulatedQuadrupole(Element):

    def __init__(self, coord=None, alpha_N=(), beta_N=(),
            only_phase_shift=False, v_eta__omegas=None):

        self.alpha_N = alpha_N
        self.beta_N = beta_N
        self.coord = coord
        self.only_phase_shift = only_phase_shift

        if only_phase_shift:
            assert(v_eta__omegas is not None)
            self.v_eta__omegas = v_eta__omegas
            C_bar_N = np.zeros_like(alpha_N)
            for nn in range(len(C_bar_N)):
                if nn == 0:
                    C_bar_N[nn] = 2*np.pi
                    continue
                if nn == 1:
                    C_bar_N[nn] = 0.
                    continue
                C_bar_N[nn] = (nn-1.)/nn * C_bar_N[nn-2]
            S_bar_N = np.zeros_like(beta_N)
            for nn in range(len(S_bar_N)):
                if nn == 0:
                    S_bar_N[nn] = 2*np.pi
                    continue
                if nn == 1:
                    S_bar_N[nn] = 0.
                    continue
                S_bar_N[nn] = (nn-1.)/nn * S_bar_N[nn-2]
            self.C_bar_N = C_bar_N
            self.S_bar_N = S_bar_N


    def track(self, bunch):
        kpart = np.zeros_like(bunch.z)
        for pp, a_pp in enumerate(self.alpha_N):
            kpart += a_pp * bunch.z**pp
        for pp, b_pp in enumerate(self.beta_N):
            kpart += b_pp * bunch.dp**pp

        if self.only_phase_shift:
            r_part = np.sqrt(bunch.z**2 + (self.v_eta__omegas * bunch.dp)**2)
            for pp, a_pp in enumerate(self.alpha_N):
                kpart -= a_pp * self.C_bar_N[pp] / (2*np.pi) * r_part**pp
            for pp, b_pp in enumerate(self.beta_N):
                kpart -= (b_pp /(self.v_eta__omegas)**pp * self.S_bar_N[pp]
                        / (2*np.pi) * r_part**pp)

        dp_part = kpart * getattr(bunch, self.coord)
        p_old = getattr(bunch, self.coord + 'p')
        setattr(bunch, self.coord + 'p', p_old + dp_part)
