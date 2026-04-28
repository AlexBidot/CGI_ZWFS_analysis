import numpy as np
import roman_preflight_proper
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import pyzelda.zelda as zelda
import pyzelda.sphere.sequence as zseq
import scipy.ndimage as ndimage

from pathlib import Path
from tqdm import tqdm
from astropy.io import fits
from pyzelda.utils import aperture
from pyzelda.utils import zernike

from corgisim import outputs, scene, instrument, jitter

try:
    import CGI_ZWFS_analysis.simulate_zwfs as zwfs
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).parent.parent))
    import CGI_ZWFS_analysis.simulate_zwfs as zwfs

bandpass_values = {
    '1F': {'wave': 575e-9, 'bandwidth': 0.101},
    '1A': {'wave': 556e-9, 'bandwidth': 0.035},
    '1B': {'wave': 575e-9, 'bandwidth': 0.033},
    '1C': {'wave': 594e-9, 'bandwidth': 0.032},
    }


if __name__ == '__main__':
    __spec__ = None

    # Copy the default prescription file
    roman_preflight_proper.copy_here()

    root = Path('/Users/avigan/data/Roman/ZWFS/')

    vmag   = 2
    sptype = 'G0V'

    bandpass = '1B'
    dm_case  = 'flat'   # flat, 3e-8, 5e-9, 2e-9

    generate = False
    analyze  = True

    # paths
    path = root / 'jitter' / f'dm={dm_case}_bandpass={bandpass}'
    path_raw = path / 'raw'
    path_processed = path / 'processed'

    path_processed.mkdir(parents=True, exist_ok=True)

    if generate:
        ref_star_properties = {
            'Vmag': vmag, 'spectral_type': sptype, 'magtype': 'vegamag',
            }

        # jitter keywords
        nrings   = 7
        r_ring0  = 0.075
        noffsets = np.array([3, 5, 7, 10, 10, 15, 20])
        dr_rings = np.array([0.3, 0.3, 0.3, 0.8, 1.6, 3.2, 6.4])
        jitter_keywords = {
            'add_jitter': 1,
            'jitter_sigmax': 4.0,
            'jitter_sigmay': 4.0,
            'N_rings_of_offsets': nrings,
            'N_offsets_per_ring': noffsets,
            'starting_offset_ang_by_ring': np.arange(nrings)*45 % 180,
            'r_ring0': r_ring0,
            'dr_rings': dr_rings,
            'outer_radius_of_offset_circle': r_ring0 + dr_rings.sum()
            }

        # x_offsets, y_offsets, A_offsets, x_outer_dict, yu_outer_dict, yl_outer_dict, boundary_coords_dict = jitter.Determine_offsets_and_areas(
        #     jitter_keywords['outer_radius_of_offset_circle'],
        #     jitter_keywords['N_rings_of_offsets'],
        #     jitter_keywords['N_offsets_per_ring'],
        #     jitter_keywords['starting_offset_ang_by_ring'],
        #     jitter_keywords['r_ring0'],
        #     jitter_keywords['dr_rings'])

        # # Step 2: Use jitter.Plot_ALL_Offsets_And_Region_Outlines to draw the figure
        # jitter.Plot_ALL_Offsets_And_Region_Outlines(
        #     x_offsets,y_offsets,x_outer_dict,
        #     yu_outer_dict,
        #     yl_outer_dict,
        #     boundary_coords_dict,
        #     jitter_keywords['N_rings_of_offsets'],
        #     jitter_keywords['N_offsets_per_ring'])

        #%% simulate data

        # generate clear pupil and ZWFS images without jitter
        scene = zwfs.get_zwfs_data(ref_star_properties, zwfs.PupilType.CLEAR, bandpass=bandpass, dm_case=dm_case)

        outpath = path_raw / 'pupil=clear_jitter=0.fits'
        outputs.save_hdu_to_fits(scene, outdir=outpath.parent, filename=outpath.name, write_as_L1=False, overwrite=True)

        scene = zwfs.get_zwfs_data(ref_star_properties, zwfs.PupilType.ZWFS, bandpass=bandpass, dm_case=dm_case)

        outpath = path_raw / 'pupil=zwfs_jitter=0.fits'
        outputs.save_hdu_to_fits(scene, outdir=outpath.parent, filename=outpath.name, write_as_L1=False, overwrite=True)

        # generate clear pupil and ZWFS images with jitter
        scene = zwfs.get_zwfs_data(ref_star_properties, zwfs.PupilType.CLEAR, bandpass=bandpass, dm_case=dm_case, jitter_keywords=jitter_keywords)

        outpath = path_raw / 'pupil=clear_jitter=1.fits'
        outputs.save_hdu_to_fits(scene, outdir=outpath.parent, filename=outpath.name, write_as_L1=False, overwrite=True)

        scene = zwfs.get_zwfs_data(ref_star_properties, zwfs.PupilType.ZWFS, bandpass=bandpass, dm_case=dm_case, jitter_keywords=jitter_keywords)

        outpath = path_raw / 'pupil=zwfs_jitter=1.fits'
        outputs.save_hdu_to_fits(scene, outdir=outpath.parent, filename=outpath.name, write_as_L1=False, overwrite=True)

    if analyze:
        opd_map_size = 296
        pupil_size   = 292

        # no jitter
        zelda_pupil_file = 'pupil=zwfs_jitter=0'
        clear_pupil_file = 'pupil=clear_jitter=0'
        dark_file        = None

        z = zelda.Sensor('ROMAN-CGI')
        clear_pupil, zelda_pupil, center = z.read_files(path_raw, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=True, center=(175.5, 175.5), shift_method='interp')

        wave = bandpass_values[bandpass]['wave']
        opd_no_jitter = z.analyze(clear_pupil, zelda_pupil, wave=wave)

        # no jitter
        zelda_pupil_file = 'pupil=zwfs_jitter=1'
        clear_pupil_file = 'pupil=clear_jitter=1'
        dark_file        = None

        z = zelda.Sensor('ROMAN-CGI')
        clear_pupil, zelda_pupil, center = z.read_files(path_raw, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=True, center=(175.5, 175.5), shift_method='interp')

        wave = bandpass_values[bandpass]['wave']
        opd_jitter = z.analyze(clear_pupil, zelda_pupil, wave=wave)

        # erode pupil to avoid edge effects
        pupil = (opd_no_jitter != 0)
        pupil_analysis = ndimage.binary_erosion(pupil, iterations=4)

        # subtract piston and save
        opd_no_jitter[pupil_analysis == 0] = np.nan
        opd_no_jitter = opd_no_jitter - np.nanmean(opd_no_jitter[pupil_analysis])
        fits.writeto(path_processed / 'opd_map_jitter=0.fits', opd_no_jitter, overwrite=True)

        opd_jitter[pupil_analysis == 0] = np.nan
        opd_jitter = opd_jitter - np.nanmean(opd_jitter[pupil_analysis])
        fits.writeto(path_processed / 'opd_map_jitter=1.fits', opd_jitter, overwrite=True)

        #%% tip-tilt subtraction

        # build custom Zernike basis
        nzernike = 20
        rho, theta = aperture.coordinates(opd_map_size, pupil_size/2, cpix=True, strict=False, outside=0)
        basis = zernike.arbitrary_basis(pupil_analysis, nterms=nzernike, rho=rho, theta=theta)

        zern_basis = np.nan_to_num(basis)
        zern_basis_vec  = np.reshape(zern_basis, (nzernike, -1))
        zern_basis_mask = pupil_analysis.flatten()
        zern_basis_rms2pv = np.zeros((nzernike))

        ipup = pupil_analysis != 0
        for i in range(nzernike):
            mode = zern_basis[i, ipup]
            zern_basis_rms2pv[i] = (mode.max() - mode.min()) / mode.std()

        # subtraction
        pupil = pupil_analysis
        basis = zern_basis_vec
        mask  = zern_basis_mask

        opd_maps = np.stack((opd_no_jitter, opd_jitter))
        opd_maps_nott = np.zeros_like(opd_maps)
        for iopd, opd in enumerate(opd_maps):
            # projection on Zernike basis
            data = np.reshape(opd*pupil, (1, -1))
            data[:, mask == 0] = 0
            zcoeff = (basis @ data.T).squeeze() / mask.sum()

            # tip, tilt
            tip_nm_rms  = zcoeff[1]
            tilt_nm_rms = zcoeff[2]

            opd_no_tt = opd - tip_nm_rms * zern_basis[1] - tilt_nm_rms * zern_basis[2]
            opd_maps_nott[iopd] = opd_no_tt

        #%%
        opd_diff = opd_jitter - opd_no_jitter
        opd_diff_nott = opd_maps_nott[1] - opd_maps_nott[0]

        fig = plt.figure('Reconstruction error', figsize=(9, 7))
        fig.clf()

        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.1])

        cmap = mpl.cm.bwr
        norm = colors.Normalize(vmin=-1, vmax=1)

        ax = fig.add_subplot(gs[0])
        ax.set_rasterization_zorder(-1_000)

        cim = ax.imshow(opd_diff, cmap=cmap, norm=norm)

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        ax.set_title(f'bandpass={bandpass}, DM={dm_case}')

        ax = fig.add_subplot(gs[1])
        cbar = fig.colorbar(cim, cax=ax)
        cbar.set_label('Reconstruction error [nm]')

        fig.subplots_adjust(left=0.02, right=0.85, bottom=0.03, top=0.94, wspace=0.1)

        fig.savefig(path_processed / f'reconstruction_error_DM={dm_case}.png', dpi=300)

        #%% PSD
        psd_cutoff = 100  # cycles/pupil

        psd_cube = zseq.compute_psd(None, np.nan_to_num(opd_diff[np.newaxis, ...]), freq_cutoff=psd_cutoff, return_fft=False, pupil_mask=pupil_analysis)
        psd_int, psd_bnds = zseq.integrate_psd(None, psd_cube, freq_cutoff=psd_cutoff)

        psd_cube = zseq.compute_psd(None, np.nan_to_num(opd_diff_nott[np.newaxis, ...]), freq_cutoff=psd_cutoff, return_fft=False, pupil_mask=pupil_analysis)
        psd_int_nott, psd_bnds_nott = zseq.integrate_psd(None, psd_cube, freq_cutoff=psd_cutoff)

        fig = plt.figure('Reconstruction error PSD', figsize=(9, 7))
        fig.clf()

        ax = fig.add_subplot(111)

        ax.step(psd_bnds[:, 0], psd_int, label='')
        ax.step(psd_bnds[:, 0], psd_int_nott, label='Tip-tilt removed')

        ax.set_xlabel('Spatial frequency [c/p]')
        ax.set_xlim(0, 60)

        ax.set_yscale('log')
        ax.set_ylabel('PSD [nm rms / (c/p)]')
        ax.set_ylim(1e-2, 1e-1)

        ax.legend()

        ax.set_title(f'bandpass={bandpass}, DM={dm_case}')

        fig.subplots_adjust(left=0.18, right=0.95, bottom=0.11, top=0.94, wspace=0.1)

        fig.savefig(path_processed / f'reconstruction_error_psd_DM={dm_case}.png', dpi=300)
