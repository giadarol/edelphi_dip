import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import PyECLOUD.mystyle as ms
import PyECLOUD.myfilemanager as mfm

beta_func = 92.7
T_rev = 88.9e-6
q_frac = .27
Qs = 4.9e-3
l_min = -6
l_max = 4
alpha_0 = -1.61237838e-03
min_strength = 0
max_strength = 1.6
max_strength_tau_plot = 1.6
tau_min = 0
tau_max = 300
flag_mode_0 = False
factor_DQ0 = 0.85
DQ_0 = -alpha_0 * beta_func/4/np.pi*factor_DQ0


# Comparison for paper
flag_mode_unstab = False
dict_plot = {
        't1':  {'fname':'./processed_data/compact_t1_fit.mat', 'tilt_lines':False, 'scale_x':1, 'label':'t1'},
        't2': {'fname':'./processed_data/compact_t2_fit.mat', 'tilt_lines':False, 'scale_x':1, 'label':'t2'},
        't3': {'fname':'./processed_data/compact_t3_fit.mat', 'tilt_lines':True, 'scale_x':1, 'label':'t3'},
       }


# flag_mode_unstab = True
# dict_plot = {
#          'pic':{'fname':'./processed_data/compact_pic_fit.mat', 'tilt_lines':True, 'scale_x':1., 'insta_thresh': 1.23, 'label': 'Particle In Cell'},
#          't6': {'fname':'./processed_data/compact_t6_fit.mat', 'tilt_lines':True, 'scale_x':1, 'insta_thresh': 1.42, 'label': r'$\Delta$Q$_\Phi\neq$0, $\Delta$Q$_R\neq$0'+'\n+ transverse non-linear map'},
#          }

colorlist = ['b', 'r', 'g', 'orange', 'k']
colorlist = ['C3', 'g']
colorlist = None



def extract_independent_lines(strength_list,
        all_freqs, all_aps, min_dist, n_indep_list,
        allowed_range=None):

    all_freq_indep = []
    all_aps_indep = []
    all_stre_indep = []
    for jjj, sss in enumerate(strength_list):
        this_freqs = np.abs(all_freqs[jjj, :])
        this_aps = np.abs(all_aps[jjj, :])

        i_sorted = np.argsort(this_aps)[::-1]

        this_f_indep = [0]
        this_ap_indep = [0]
        this_stren_indep = [0]
        for ifr in i_sorted:
            if len(this_f_indep) > n_indep_list[jjj]:
                break
            ff = this_freqs[ifr]
            if allowed_range is not None:
                if ff>allowed_range[1] or ff<allowed_range[0]:
                    continue
            if np.min(np.abs(ff - np.array(this_f_indep))) > min_dist:
                this_f_indep.append(ff)
                this_ap_indep.append(this_aps[ifr])
                this_stren_indep.append(sss)

        all_freq_indep += this_f_indep[1:]
        all_aps_indep += this_ap_indep[1:]
        all_stre_indep += this_stren_indep[1:]

    return all_freq_indep, all_aps_indep, all_stre_indep


plt.close('all')
ms.mystyle_arial(fontsz=14, dist_tick_lab=5, traditional_look=False)
fig1 = plt.figure(1, figsize=(6.4*1.2, 4.8))
ax1 = fig1.add_subplot(111)
axshare = None
figharm_list = []
figintra_list = []
for ii, ll in enumerate(dict_plot.keys()):
    oo = mfm.myloadmat_to_obj(dict_plot[ll]['fname'])
    if flag_mode_unstab:
        insta_thresh = dict_plot[ll]['insta_thresh']
    tilt_lines = dict_plot[ll]['tilt_lines']
    scale_x = dict_plot[ll]['scale_x']
    kwargs = {}
    if colorlist is not None:
        kwargs['color'] = colorlist[ii]
    # ax1.plot(oo.strength_list*scale_x, oo.p_list_centroid/T_rev, label=ll,
    #         linewidth=2, **kwargs)
    ax1.plot(oo.strength_list, oo.p_list_centroid/T_rev, '.', alpha=.5,
        markeredgewidth=0, **kwargs)
    from scipy.signal import savgol_filter
    mask_plot = oo.strength_list < max_strength_tau_plot
    smooth_gr = savgol_filter(oo.p_list_centroid[mask_plot]/T_rev, 31, 5)
    ax1.plot(oo.strength_list[mask_plot], smooth_gr,
            label=dict_plot[ll]['label'],
            linestyle='--', linewidth=3, **kwargs)

    mask_strength = (oo.strength_list <= max_strength)

    # Centrois sussix spectrogram
    ap_list = oo.ap_list
    N_lines = ap_list.shape[1]
    strength_list = oo.strength_list[mask_strength]*scale_x
    freq_list = oo.freq_list[mask_strength, :]

    figharm = plt.figure(100+ii)
    maxsize = np.max(np.array(ap_list))
    axharm = figharm.add_subplot(111, sharex=axshare, sharey=axshare)
    axshare = axharm
    str_mat = np.dot(np.atleast_2d(np.ones(N_lines)).T,
            np.atleast_2d(np.array(strength_list)))
    # for lll in range(l_min-10, l_max+10):
    #     axharm.plot(strength_list, lll + float(tilt_lines)*DQ_0*strength_list/Qs,
    #             alpha=0.5, linestyle='-', color='grey')
    axharm.scatter(x=str_mat.flatten(),
            y=(np.abs(np.array(freq_list)).T.flatten()-q_frac)/Qs,
            s=np.clip(np.array(ap_list).T.flatten()/maxsize*10, 0.0, 10),
            color='darkblue')

    all_freq_indep_0, all_aps_indep_0, all_stre_indep_0 = extract_independent_lines(
        strength_list, np.abs(np.array(freq_list)), np.array(ap_list),
        min_dist=3e-3, n_indep_list=np.zeros_like(strength_list, dtype=np.int)+2)
    indep_normalized_0 = (np.array(all_freq_indep_0)-q_frac)/Qs
    mask_keep_0 = np.abs(indep_normalized_0)<1.5
    axharm.plot(np.array(all_stre_indep_0)[mask_keep_0],
                indep_normalized_0[mask_keep_0], '.', color='C03')

    freq_mode_0, ap_mode_0, stre_mode_0 = extract_independent_lines(
        strength_list, np.abs(np.array(freq_list)), np.array(ap_list),
        min_dist=3e-3, n_indep_list=np.zeros_like(strength_list, dtype=np.int)+1,
        allowed_range=(q_frac, q_frac + .8*Qs))

    axharm.set_ylim(l_min, l_max)
    axharm.set_xlim(min_strength, max_strength)
    figharm.suptitle(ll)
    figharm.subplots_adjust(right=.83)
    figharm_list.append(figharm)

    # Plot data from intrabunch motion
    all_freqs = np.concatenate(
            (oo.freqs_1mode_re_list[mask_strength],
             oo.freqs_1mode_im_list[mask_strength]), axis=1)
    all_aps = np.concatenate(
            (oo.ap_1mode_re_list[mask_strength],
             oo.ap_1mode_im_list[mask_strength]), axis=1)
    # Renorm to each colunms
    for jjj, sss in enumerate(strength_list):
        all_aps[jjj, :] /= np.mean(all_aps[jjj, all_aps[jjj, :]>0])
    maxsizeintra = np.max(np.array(all_aps))
    figintra = plt.figure(200+ii)
    axintra = figintra.add_subplot(111, sharex=axshare, sharey=axshare)
    str_mat_intra = np.dot(np.atleast_2d(np.ones(all_freqs.shape[1])).T,
            np.atleast_2d(np.array(strength_list)))
    scale_marker = 1.5
    axintra.scatter(x=str_mat_intra.flatten(),
            y=(np.abs(np.array(all_freqs)).T.flatten()-q_frac)/Qs,
            s=np.clip(np.array(all_aps).T.flatten()/maxsizeintra*scale_marker,
                0.0, scale_marker),
            #c=np.clip(np.array(all_aps).T.flatten()/maxsizeintra, 0.3, 0.4),
            cmap=cm.Blues, norm=Normalize(vmin=0, vmax=0.5),
            color='C0')

    if flag_mode_0:
        # Plot mode zero
        mask_plot_mode_0 = np.array(stre_mode_0) < insta_thresh
        axintra.plot(np.array(stre_mode_0)[mask_plot_mode_0],
                (np.array(freq_mode_0)[mask_plot_mode_0]-q_frac)/Qs, '.k')
    if flag_mode_unstab:
        # Plot unstable freq
        freq_instab, ap_instab, stre_instab = extract_independent_lines(
            strength_list, all_freqs, all_aps, 1e-3,
            np.ones_like(strength_list, dtype=np.int))
        mask_instab = np.array(stre_instab) > insta_thresh
        axintra.plot(np.array(stre_instab)[mask_instab],
                (np.array(freq_instab)[mask_instab]-q_frac)/Qs, '.',
                color='C3')

    axintra.set_yticks(np.arange(l_min+1, l_max-0.2))
    axintra.grid(axis='y', linestyle='--')
    axintra.set_xlabel('e-cloud strength')
    axintra.set_ylabel(r'(Q - Q$_0$)/Q$_s$')
    figintra.suptitle(ll)
    figintra.subplots_adjust(bottom=.12, right=.85)
    figintra_list.append(figintra)

    # min_dist = 3e-3
    # n_indep_list = np.zeros_like(strength_list, dtype=np.int) + 2
    # n_indep_list[oo.n_sample_list<1000] = 1
    # all_freq_indep, all_aps_indep, all_stre_indep = extract_independent_lines(
    #     strength_list, all_freqs, all_aps, min_dist, n_indep_list)
    # min_dist = 3e-3
    # n_indep_list = np.zeros_like(strength_list, dtype=np.int) + 2
    # n_indep_list[oo.n_sample_list<1000] = 1
    # all_freq_indep, all_aps_indep, all_stre_indep = extract_independent_lines(
    #     strength_list, all_freqs, all_aps, min_dist, n_indep_list)

    # indep_normalized = (np.array(all_freq_indep)-q_frac)/Qs
    # mask_keep = np.abs(indep_normalized)<2
    # axintra.plot(np.array(all_stre_indep)[mask_keep],
    #             indep_normalized[mask_keep], '.', color='C03')

    # for i_sb in [-1, 1.1]:
    #     freq_sb, ap_sb, stre_sb = extract_independent_lines(
    #         strength_list, np.abs(np.array(freq_list)), np.array(ap_list),
    #         min_dist=3e-3, n_indep_list=np.zeros_like(strength_list, dtype=np.int)+1,
    #         allowed_range=(q_frac+(i_sb*Qs-0.1*Qs), q_frac + (i_sb + 1.)*Qs))
    #     sb_normalized = (np.array(freq_sb)-q_frac)/Qs
    #     axintra.plot(np.array(stre_sb), sb_normalized, '.', color='C01')

ax1.legend(loc='upper left', fontsize='medium', frameon=False)
#ax1.grid(True, linestyle=':')
ax1.set_xlim(min_strength, max_strength)
ax1.set_ylim(tau_min, tau_max)
ax1.set_xlabel('e-cloud strength')
ax1.set_ylabel(r'Instability growth rate [s$^{-1}$]')
fig1.subplots_adjust(right=.71, bottom=.12, top=.85)
plt.show()
