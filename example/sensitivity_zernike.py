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
import proper

from pathlib import Path
from tqdm import tqdm
from astropy.io import fits
from pyzelda.utils import aperture, zernike, imutils

from corgisim import outputs, scene, instrument, jitter

try:
    import CGI_ZWFS_analysis.simulate_zwfs as zwfs
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).parent.parent))
    import CGI_ZWFS_analysis.simulate_zwfs as zwfs


if __name__ == '__main__':
    __spec__ = None

    # Copy the default prescription file
    roman_preflight_proper.copy_here()

    root = Path('/Users/avigan/data/Roman/ZWFS/')

    vmag   = 2
    sptype = 'G0V'

    bandpass = '1F'
    dm_case  = 'flat'   # flat, 3e-8, 5e-9, 2e-9

    Zidx    = 159     # Zernike polynomial Noll index - test with 620 works
    Zamp    = 0.1    # nm rms
    emgain  = 1
    exptime = 2
    nexp    = 150

    generate = False
    observe  = False
    analyze  = True
    merge    = False

    # paths
    path = root / 'sensitivity_zernike' / f'dm={dm_case}_bandpass={bandpass}_z={Zidx}_ampl={Zamp:0.3f}'
    path_raw = path / 'raw'
    path_processed = path / 'processed'

    path_raw.mkdir(parents=True, exist_ok=True)
    path_processed.mkdir(parents=True, exist_ok=True)

    Tint = exptime * nexp

    if generate:
        #%% simulate data
        ref_star_properties = {
            'Vmag': vmag, 'spectral_type': sptype, 'magtype': 'vegamag',
            }

        zernike_keywords = {
            'zindex': np.array([Zidx, ]),
            'zval_m': np.array([Zamp, ])*1e-9,
            }

        # generate noiseless clear pupil and ZWFS images
        for pupil_type in (zwfs.PupilType.CLEAR, zwfs.PupilType.ZWFS, ):
        #for pupil in (zwfs.PupilType.ZWFS, ):

            # noiseless reference data
            noiseless_ref = zwfs.generate_zwfs_data(ref_star_properties, pupil_type, bandpass=bandpass, dm_case=dm_case)

            outpath = path_raw / f'pupil={pupil_type.value}_aberration=0_emccd=0.fits'
            fits.writeto(outpath, noiseless_ref.host_star_image.data, overwrite=True)
            with fits.open(outpath, mode='update') as hdul:
                hdul[0].header['ZINDEX']  = (Zidx, 'Aberration Zernike index')
                hdul[0].header['ZAMPL']   = (Zamp, '[nm rms] Aberration amplitude')

            # noiseless data with differential aberration
            noiseless = zwfs.generate_zwfs_data(ref_star_properties, pupil_type, bandpass=bandpass, dm_case=dm_case, optics_keywords=zernike_keywords,)

            outpath = path_raw / f'pupil={pupil_type.value}_aberration=1_emccd=0.fits'
            fits.writeto(outpath, noiseless.host_star_image.data, overwrite=True)
            with fits.open(outpath, mode='update') as hdul:
                hdul[0].header['ZINDEX']  = (Zidx, 'Aberration Zernike index')
                hdul[0].header['ZAMPL']   = (Zamp, '[nm rms] Aberration amplitude')

    if observe:
        #%% generate EMCCD-observed clear pupil and ZWFS images
        for pupil_type in (zwfs.PupilType.CLEAR, zwfs.PupilType.ZWFS, ):
            # reference data
            inpath = path_raw / f'pupil={pupil_type.value}_aberration=0_emccd=0.fits'
            noiseless_ref = fits.getdata(inpath)

            scene = zwfs.DummyScene(noiseless_ref)
            exposures_ref = zwfs.observe_with_emccd(scene, emgain=emgain, photon_counting=False, exposure_time=exptime, num_exposures=nexp)
            data = np.array([exp.data.astype(float) for exp in exposures_ref])
            exposure_ref = np.mean(data, axis=0)

            outpath = path_raw / f'pupil={pupil_type.value}_aberration=0_emccd=1_Tint={Tint}.fits'
            fits.writeto(outpath, exposure_ref, overwrite=True)
            with fits.open(outpath, mode='update') as hdul:
                hdul[0].header['ZINDEX']  = (Zidx, 'Aberration Zernike index')
                hdul[0].header['ZAMPL']   = (Zamp, '[nm rms] Aberration amplitude')
                hdul[0].header['EMGAIN']  = (emgain, 'EMCCD gain')
                hdul[0].header['EXPTIME'] = (exptime, '[s] Integration time of individual exposures')
                hdul[0].header['NEXP']    = (nexp, 'Number of exposures')

            # data with differential aberration
            inpath = path_raw / f'pupil={pupil_type.value}_aberration=1_emccd=0.fits'
            noiseless = fits.getdata(inpath)

            scene = zwfs.DummyScene(noiseless)
            exposures = zwfs.observe_with_emccd(scene, emgain=emgain, photon_counting=False, exposure_time=exptime, num_exposures=nexp)
            data = np.array([exp.data.astype(float) for exp in exposures])
            exposure = np.mean(data, axis=0)

            outpath = path_raw / f'pupil={pupil_type.value}_aberration=1_emccd=1_Tint={Tint}.fits'
            fits.writeto(outpath, exposure, overwrite=True)
            with fits.open(outpath, mode='update') as hdul:
                hdul[0].header['ZINDEX']  = (Zidx, 'Aberration Zernike index')
                hdul[0].header['ZAMPL']   = (Zamp, '[nm rms] Aberration amplitude')
                hdul[0].header['EMGAIN']  = (emgain, 'EMCCD gain')
                hdul[0].header['EXPTIME'] = (exptime, '[s] Integration time of individual exposures')
                hdul[0].header['NEXP']    = (nexp, 'Number of exposures')

        #%%
        fig = plt.figure('ZWFS images', figsize=(20.5, 7))
        fig.clf()

        ax = fig.add_subplot(131)
        ax.imshow(exposure)
        ax.set_title('ZWFS image - noiseless')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        ax = fig.add_subplot(132, sharex=ax, sharey=ax)
        ax.imshow(noiseless - noiseless_ref, vmin=-10, vmax=10, cmap='bwr')
        ax.set_title('Differential ZWFS image - noiseless')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        ax = fig.add_subplot(133, sharex=ax, sharey=ax)
        ax.imshow(exposure - exposure_ref, vmin=-10, vmax=10, cmap='bwr')
        ax.set_title('Differential ZWFS image')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.94, wspace=0.01)
        fig.savefig(path_processed / 'zwfs_images.png', dpi=300)

    #%%
    if analyze:
        plt.close('all')

        opd_map_size = 296
        pupil_size   = 292

        # reference case - NOISELESS
        zelda_pupil_file = f'pupil=zwfs_aberration=0_emccd=0'
        clear_pupil_file = f'pupil=clear_aberration=0_emccd=0'
        dark_file        = None

        z = zelda.Sensor('ROMAN-CGI')
        clear_pupil, zelda_pupil, center = z.read_files(path_raw, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=True, center=(175.5, 175.5), shift_method='interp')

        wave = zwfs.bandpass_values[bandpass]['wave']
        opd_ref_noiseless = z.analyze(clear_pupil, zelda_pupil, wave=wave)

        # reference case
        zelda_pupil_file = f'pupil=zwfs_aberration=0_emccd=1_Tint={Tint}'
        clear_pupil_file = f'pupil=clear_aberration=0_emccd=1_Tint={Tint}'
        dark_file        = None

        z = zelda.Sensor('ROMAN-CGI')
        clear_pupil, zelda_pupil, center = z.read_files(path_raw, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=True, center=(175.5, 175.5), shift_method='interp')

        wave = zwfs.bandpass_values[bandpass]['wave']
        opd_ref = z.analyze(clear_pupil, zelda_pupil, wave=wave)

        # differential aberration case - NOISELESS
        zelda_pupil_file = f'pupil=zwfs_aberration=1_emccd=0'
        clear_pupil_file = f'pupil=clear_aberration=1_emccd=0'
        dark_file        = None

        z = zelda.Sensor('ROMAN-CGI')
        clear_pupil, zelda_pupil, center = z.read_files(path_raw, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=True, center=(175.5, 175.5), shift_method='interp')

        wave = zwfs.bandpass_values[bandpass]['wave']
        opd_noiseless = z.analyze(clear_pupil, zelda_pupil, wave=wave)

        # differential aberration case
        zelda_pupil_file = f'pupil=zwfs_aberration=1_emccd=1_Tint={Tint}'
        clear_pupil_file = f'pupil=clear_aberration=1_emccd=1_Tint={Tint}'
        dark_file        = None

        z = zelda.Sensor('ROMAN-CGI')
        clear_pupil, zelda_pupil, center = z.read_files(path_raw, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=True, center=(175.5, 175.5), shift_method='interp')

        wave = zwfs.bandpass_values[bandpass]['wave']
        opd = z.analyze(clear_pupil, zelda_pupil, wave=wave)

        #%% additional post-processing

        # erode pupil to avoid edge effects
        pupil = (opd_ref != 0)
        pupil_analysis = ndimage.binary_erosion(pupil, iterations=4)

        # subtract piston and save
        opd_ref_noiseless[pupil_analysis == 0] = np.nan
        opd_ref_noiseless = opd_ref_noiseless - np.nanmean(opd_ref_noiseless[pupil_analysis])
        fits.writeto(path_processed / 'opd_map_ref_noiseless.fits', opd_ref_noiseless, overwrite=True)

        opd_noiseless[pupil_analysis == 0] = np.nan
        opd_noiseless = opd_noiseless - np.nanmean(opd_noiseless[pupil_analysis])
        fits.writeto(path_processed / 'opd_map_noiseless.fits', opd_noiseless, overwrite=True)

        opd_ref[pupil_analysis == 0] = np.nan
        opd_ref = opd_ref - np.nanmean(opd_ref[pupil_analysis])
        fits.writeto(path_processed / f'opd_map_ref_Tint={Tint}.fits', opd_ref, overwrite=True)

        opd[pupil_analysis == 0] = np.nan
        opd = opd - np.nanmean(opd[pupil_analysis])
        fits.writeto(path_processed / f'opd_map_Tint={Tint}.fits', opd, overwrite=True)

        #%%
        opd_diff_noiseless = opd_noiseless - opd_ref_noiseless

        opd_diff = opd - opd_ref
        opd_diff = imutils.sigma_filter(opd_diff, box=12, nsigma=3, iterate=True)

        data = np.stack((opd_diff_noiseless, opd_diff))
        fits.writeto(path_processed / f'differential_opd_maps_Tint={Tint}.fits', data, overwrite=True)

        # plot
        plt.close('all')
        fig = plt.figure('Reconstruction error', figsize=(9, 7))
        fig.clf()

        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.1])

        cmap = mpl.cm.bwr
        norm = colors.Normalize(vmin=-0.5, vmax=0.5)

        ax = fig.add_subplot(gs[0])
        ax.set_rasterization_zorder(-1_000)

        cim = ax.imshow(opd_diff, cmap=cmap, norm=norm)

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        #ax.set_title(f'bandpass={bandpass}, DM={dm_case}, Z$_\\mathrm{{index}}$={Zidx}, Z$_\\mathrm{{ampl}}$={Zamp*1000:.0f}pm')
        ax.set_title(f'Z$_\\mathrm{{index}}$={Zidx}, Z$_\\mathrm{{ampl}}$={Zamp*1000:.0f}pm, T$_\\mathrm{{int}}$={Tint} s')

        ax = fig.add_subplot(gs[1])
        cbar = fig.colorbar(cim, cax=ax)
        cbar.set_label('Reconstruction error [nm]')

        fig.subplots_adjust(left=0.02, right=0.84, bottom=0.03, top=0.94, wspace=0.1)

        fig.savefig(path_processed / f'differential_opd_map_z={Zidx}_ampl={Zamp:.3f}_Tint={Tint}.pdf', dpi=300)

        #%% PSD
        psd_cutoff = 150  # cycles/pupil

        # original noiseless OPD map
        psd_cube = zseq.compute_psd(None, np.nan_to_num(opd_ref_noiseless[np.newaxis, ...]), freq_cutoff=psd_cutoff, return_fft=False, pupil_mask=pupil_analysis)
        psd_int_ref_noiseless, psd_bnds_ref_noiseless = zseq.integrate_psd(None, psd_cube, freq_cutoff=psd_cutoff)

        # differential OPD maps
        psd_cube = zseq.compute_psd(None, np.nan_to_num(opd_diff_noiseless[np.newaxis, ...]), freq_cutoff=psd_cutoff, return_fft=False, pupil_mask=pupil_analysis)
        psd_int_diff_noiseless, psd_bnds_diff_noiseless = zseq.integrate_psd(None, psd_cube, freq_cutoff=psd_cutoff)

        psd_cube = zseq.compute_psd(None, np.nan_to_num(opd_diff[np.newaxis, ...]), freq_cutoff=psd_cutoff, return_fft=False, pupil_mask=pupil_analysis)
        psd_int_diff, psd_bnds_diff = zseq.integrate_psd(None, psd_cube, freq_cutoff=psd_cutoff)

        #%% plot
        fig = plt.figure('Reconstruction error PSD', figsize=(9, 7))
        fig.clf()

        ax = fig.add_subplot(111)

        ax.step(psd_bnds_ref_noiseless[:, 0], psd_int_ref_noiseless, label='Full OPD (noiseless)')
        ax.step(psd_bnds_diff_noiseless[:, 0], psd_int_diff_noiseless, label='Differential OPD (noiseless)')
        ax.step(psd_bnds_diff[:, 0], psd_int_diff, label='Differential OPD (EMCCD)')

        ax.set_xlabel('Spatial frequency [c/p]')
        ax.set_xlim(0, 60)

        ax.set_yscale('log')
        ax.set_ylabel('PSD [nm rms / (c/p)]')
        ax.set_ylim(1e-4, 1e-1)

        ax.legend(fontsize='small', loc='upper right')

        ax.set_title(f'bandpass={bandpass}, DM={dm_case}, Z$_\\mathrm{{index}}$={Zidx}, Z$_\\mathrm{{ampl}}$={Zamp*1000:.0f}pm')

        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.11, top=0.94, wspace=0.1)

        fig.savefig(path_processed / f'differential_opd_map_psd_Tint={Tint}.png', dpi=300)

    if merge:
        Tints = [100, 300, 1000, 3000, 10_000]
        psd_cutoff = 150  # cycles/pupil

        data = fits.getdata(path_processed / f'differential_opd_maps_Tint={Tints[0]}.fits')
        opd_diff_noiseless = data[0]

        # erode pupil to avoid edge effects
        pupil = (opd_diff_noiseless != 0)
        pupil_analysis = ndimage.binary_erosion(pupil, iterations=4)

        # differential OPD maps
        psd_cube = zseq.compute_psd(None, np.nan_to_num(opd_diff_noiseless[np.newaxis, ...]), freq_cutoff=psd_cutoff, return_fft=False, pupil_mask=pupil_analysis)
        psd_int_diff_noiseless, psd_bnds_diff_noiseless = zseq.integrate_psd(None, psd_cube, freq_cutoff=psd_cutoff)

        # plot
        fig = plt.figure('Reconstruction error PSD - Tint', figsize=(9, 7))
        fig.clf()

        ax = fig.add_subplot(111)

        ax.step(psd_bnds_diff_noiseless[:, 0], psd_int_diff_noiseless, label='Noiseless')

        for Tint in Tints:
            data = fits.getdata(path_processed / f'differential_opd_maps_Tint={Tint}.fits')
            opd_diff = data[1]
            psd_cube = zseq.compute_psd(None, np.nan_to_num(opd_diff[np.newaxis, ...]), freq_cutoff=psd_cutoff, return_fft=False, pupil_mask=pupil_analysis)
            psd_int_diff, psd_bnds_diff = zseq.integrate_psd(None, psd_cube, freq_cutoff=psd_cutoff)


            ax.step(psd_bnds_diff[:, 0], psd_int_diff, label=f'Tint = {Tint} s')

        ax.set_xlabel('Spatial frequency [c/p]')
        ax.set_xlim(0, 60)

        ax.set_yscale('log')
        ax.set_ylabel('PSD [nm rms / (c/p)]')
        ax.set_ylim(1e-4, 2e-2)

        ax.legend(fontsize='small', loc='upper right')

        ax.set_title(f'bandpass={bandpass}, DM={dm_case}, Z$_\\mathrm{{index}}$={Zidx}, Z$_\\mathrm{{ampl}}$={Zamp*1000:.0f}pm')

        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.11, top=0.94, wspace=0.1)

        fig.savefig(path_processed / f'differential_opd_map_psd_Tint=all.pdf', dpi=300)
