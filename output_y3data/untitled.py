# cells, cell_weights_wide = get_cell_weights_wide(wide_data, 
#                                                  overlap_weighted_pchat=True, 
#                                                  force_assignment=False, 
#                                                  cell_key='cell_wide_unsheared')

# np.save(data_dir + 'pchat.npy', cell_weights_wide)

# _, _, hists_wide_piled = pileup(hists_wide,
#                                 zbins,
#                                 zmean,
#                                 zmax_pileup,
#                                 zbins_dz,
#                                 zmax_weight,
#                                 n_bins)

# zbins_piled, zmean_piled, hists_wide_bin_cond_piled = pileup(hists_wide_bin_cond,
#                                                              zbins,
#                                                              zmean,
#                                                              zmax_pileup,
#                                                              zbins_dz,
#                                                              zmax_weight,
#                                                              n_bins)

# #plot comparing bin_cond vs not_bin_cond with pileup
# results, deltas = redshift_histograms_stats(hists_wide_piled, 
#                                             hists_wide_bin_cond_piled, 
#                                             zbins_piled, 
#                                             'using p(z|c,bhat)')

# print(results)
# print(deltas)

# fig_wide = plot_redshift_histograms(hists_wide_piled, 
#                                     hists_wide_bin_cond_piled, 
#                                     zbins_piled, 
#                                     'Bin Conditioned vs. Not Bin Conditioned n(z)', 
#                                     'Bin Conditioned', 
#                                     max_pz = 3)

# fig_wide.savefig(data_dir + f'Y3_pzc_vs_pzcbhat_wide_{run_name}.png', dpi=300)


# #plot comparing pileup effect

# means_bc, sigmas_bc = get_mean_sigma(zmean, hists_wide_bin_cond)
# means_bc_piled, sigmas_bc_piled = get_mean_sigma(zmean_piled, hists_wide_bin_cond_piled)

# plt.figure(figsize=(16.,9.))
# colors=['blue','orange','green','red']
# for i in range(len(hists_wide)):
#     plt.fill_between(zmean, hists_wide_bin_cond[i], color= colors[i],alpha=0.3) #,label="fiducial")
#     plt.axvline(means_bc[i], linestyle='-.', color= colors[i],label=str(i)+' %.3f'%(means_bc[i]))
#     plt.plot(zmean_piled, hists_wide_bin_cond_piled[i], color= colors[i])#,label="bin conditional")
#     plt.axvline(means_bc_piled[i], linestyle='-', color= colors[i],label=str(i)+' pile-up: %.3f'%(means_bc_piled[i]) )
# plt.xlabel(r'$z$')
# plt.ylabel(r'$p(z)$')
# plt.xlim(0,3)
# plt.legend()
# plt.title('Wide n(z)')
# plt.savefig(data_dir + f'Y3_nz_newbinning_onwide_bin_cond_pileup3_{run_name}.png')

# #save
# deltas.to_pickle(data_dir + 'Y3_deltas_pzc_pzcbhat.pkl')
# results.to_pickle(data_dir + 'Y3_results_pzc_pzcbhat.pkl')
# np.save(data_dir + 'Y3_hists_wide_bin_conditionalized_pileup3_{}.npy'.format(keylabel), hists_wide_bin_cond_piled)
# save_des_nz(hists_wide_bin_cond_piled, zbins_piled, n_bins, data_dir, run_name, keylabel+'_Y3_bincond_pileup3')

# #zbins=zbins_piled
# hists_wide = hists_wide_bin_cond_piled # this is what is used moving forward.
# #END pile up