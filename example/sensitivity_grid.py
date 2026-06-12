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

    grid_ampl = 0.1     # nm rms
    emgain    = 1
    exptime   = 2
    nexp      = 210

    generate = True
    observe  = True
    analyze  = True

    # paths
    path = root / 'sensitivity_grid' / f'dm=flat_bandpass={bandpass}_ampl={grid_ampl:0.3f}'
    path_raw = path / 'raw'
    path_processed = path / 'processed'

    path_raw.mkdir(parents=True, exist_ok=True)
    path_processed.mkdir(parents=True, exist_ok=True)

#%%
    # J. Krist: *_v.fits files are in volts. The DMs use a 16 bit DAC with a 110 V range, so the smallest
    # change is about 0.0017 V. At a mean actuator gain of 3.3 nm/V, that's about 6 pm. The least significant
    # bit should be considered noisy, though.

    accuracy = 110 / (2**16)    # V/bit
    gain     = 3.3              # nm/V

    dm1_ref = proper.prop_fits_read(roman_preflight_proper.lib_dir + '/examples/hlc_flat_wfe_dm1_v.fits')
    dm2_ref = proper.prop_fits_read(roman_preflight_proper.lib_dir + '/examples/hlc_flat_wfe_dm2_v.fits')

    if generate:
        #%% simulate data
        ref_star_properties = {
            'Vmag': vmag, 'spectral_type': sptype, 'magtype': 'vegamag',
            }

        # clear pupil
        pupil_type = zwfs.PupilType.CLEAR

        clear_noiseless = zwfs.generate_zwfs_data(ref_star_properties, pupil_type, bandpass=bandpass, dm_case='flat')

        outpath = path_raw / f'pupil={pupil_type.value}_aberration=0_emccd=0.fits'
        fits.writeto(outpath, clear_noiseless.host_star_image.data, overwrite=True)

        # noiseless data with negative grid
        pupil_type = zwfs.PupilType.ZWFS

        dm1 = np.copy(dm1_ref)
        dm2 = np.copy(dm2_ref)
        dm1[0::3, 0::3] -= grid_ampl / gain
        proper.prop_fits_write(roman_preflight_proper.lib_dir + '/examples/hlc_custom_dm1_v.fits', dm1)
        proper.prop_fits_write(roman_preflight_proper.lib_dir + '/examples/hlc_custom_dm2_v.fits', dm2)

        noiseless_ref = zwfs.generate_zwfs_data(ref_star_properties, pupil_type, bandpass=bandpass, dm_case='custom')

        outpath = path_raw / f'pupil={pupil_type.value}_aberration=-1_emccd=0.fits'
        fits.writeto(outpath, noiseless_ref.host_star_image.data, overwrite=True)
        with fits.open(outpath, mode='update') as hdul:
            hdul[0].header['AMPL'] = (grid_ampl, '[nm PtV] grid amplitude')

        # noiseless data with positive grid
        dm1 = np.copy(dm1_ref)
        dm2 = np.copy(dm2_ref)
        dm1[0::3, 0::3] += grid_ampl / gain
        proper.prop_fits_write(roman_preflight_proper.lib_dir + '/examples/hlc_custom_dm1_v.fits', dm1)
        proper.prop_fits_write(roman_preflight_proper.lib_dir + '/examples/hlc_custom_dm2_v.fits', dm2)

        noiseless = zwfs.generate_zwfs_data(ref_star_properties, zwfs.PupilType.ZWFS, bandpass=bandpass, dm_case='custom')

        outpath = path_raw / f'pupil={pupil_type.value}_aberration=+1_emccd=0.fits'
        fits.writeto(outpath, noiseless.host_star_image.data, overwrite=True)
        with fits.open(outpath, mode='update') as hdul:
            hdul[0].header['AMPL'] = (grid_ampl, '[nm PtV] grid amplitude')

    if observe:
        #%% generate EMCCD-observed clear pupil and ZWFS images

        # clear pupil
        pupil_type = zwfs.PupilType.CLEAR

        inpath = path_raw / f'pupil={pupil_type.value}_aberration=0_emccd=0.fits'
        clear_noiseless = fits.getdata(inpath)

        scene = zwfs.DummyScene(clear_noiseless)
        exposures_clear = zwfs.observe_with_emccd(scene, emgain=emgain, photon_counting=False, exposure_time=exptime, num_exposures=nexp)
        data = np.array([exp.data.astype(float) for exp in exposures_clear])
        exposure_clear = np.mean(data, axis=0)

        outpath = path_raw / f'pupil={pupil_type.value}_aberration=0_emccd=1.fits'
        fits.writeto(outpath, exposure_clear, overwrite=True)
        with fits.open(outpath, mode='update') as hdul:
            hdul[0].header['EMGAIN']  = (emgain, 'EMCCD gain')
            hdul[0].header['EXPTIME'] = (exptime, '[s] Integration time of individual exposures')
            hdul[0].header['NEXP']    = (nexp, 'Number of exposures')

        # data with negative grid
        pupil_type = zwfs.PupilType.ZWFS

        inpath = path_raw / f'pupil={pupil_type.value}_aberration=-1_emccd=0.fits'
        noiseless_ref = fits.getdata(inpath)

        scene = zwfs.DummyScene(noiseless_ref)
        exposures_ref = zwfs.observe_with_emccd(scene, emgain=emgain, photon_counting=False, exposure_time=exptime, num_exposures=nexp)
        data = np.array([exp.data.astype(float) for exp in exposures_ref])
        exposure_ref = np.mean(data, axis=0)

        outpath = path_raw / f'pupil={pupil_type.value}_aberration=-1_emccd=1.fits'
        fits.writeto(outpath, exposure_ref, overwrite=True)
        with fits.open(outpath, mode='update') as hdul:
            hdul[0].header['AMPL']    = (grid_ampl, '[nm PtV] grid amplitude')
            hdul[0].header['EMGAIN']  = (emgain, 'EMCCD gain')
            hdul[0].header['EXPTIME'] = (exptime, '[s] Integration time of individual exposures')
            hdul[0].header['NEXP']    = (nexp, 'Number of exposures')

        # data with positive grid
        inpath = path_raw / f'pupil={pupil_type.value}_aberration=+1_emccd=0.fits'
        noiseless = fits.getdata(inpath)

        scene = zwfs.DummyScene(noiseless)
        exposures = zwfs.observe_with_emccd(scene, emgain=emgain, photon_counting=False, exposure_time=exptime, num_exposures=nexp)
        data = np.array([exp.data.astype(float) for exp in exposures])
        exposure = np.mean(data, axis=0)

        outpath = path_raw / f'pupil={pupil_type.value}_aberration=+1_emccd=1.fits'
        fits.writeto(outpath, exposure, overwrite=True)
        with fits.open(outpath, mode='update') as hdul:
            hdul[0].header['AMPL']    = (grid_ampl, '[nm PtV] grid amplitude')
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
        ax.imshow(noiseless - noiseless_ref, cmap='bwr')#, vmin=-300, vmax=300)
        ax.set_title('Differential ZWFS image - noiseless')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        ax = fig.add_subplot(133, sharex=ax, sharey=ax)
        ax.imshow(exposure - exposure_ref, cmap='bwr')#, vmin=-300, vmax=300
        ax.set_title('Differential ZWFS image')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.94, wspace=0.01)
        fig.savefig(path_processed / 'zwfs_images.png', dpi=300)

        #%%
        # fig = plt.figure('Values', figsize=(15, 7))
        # fig.clf()

        # ax = fig.add_subplot(111)
        # plt.plot((exposure - exposure_ref).flatten(), marker='.', linestyle='none')

        # fig.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.94)

    if analyze:
        #%%
        #plt.close('all')

        opd_map_size = 296
        pupil_size   = 292

        # negative grid - NOISELESS
        zelda_pupil_file = f'pupil=zwfs_aberration=-1_emccd=0'
        clear_pupil_file = f'pupil=clear_aberration=0_emccd=0'
        dark_file        = None

        z = zelda.Sensor('ROMAN-CGI')
        clear_pupil, zelda_pupil, center = z.read_files(path_raw, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=True, center=(175.5, 175.5), shift_method='interp')

        wave = zwfs.bandpass_values[bandpass]['wave']
        zelda_pupil_neg_noiseless = np.copy(zelda_pupil)
        opd_neg_noiseless = z.analyze(clear_pupil, zelda_pupil, wave=wave)

        # nagative grid
        zelda_pupil_file = f'pupil=zwfs_aberration=-1_emccd=1'
        clear_pupil_file = f'pupil=clear_aberration=0_emccd=1'
        dark_file        = None

        z = zelda.Sensor('ROMAN-CGI')
        clear_pupil, zelda_pupil, center = z.read_files(path_raw, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=True, center=(175.5, 175.5), shift_method='interp')

        wave = zwfs.bandpass_values[bandpass]['wave']
        zelda_pupil_neg = np.copy(zelda_pupil)
        opd_neg = z.analyze(clear_pupil, zelda_pupil, wave=wave)

        # positive grid - NOISELESS
        zelda_pupil_file = f'pupil=zwfs_aberration=+1_emccd=0'
        clear_pupil_file = f'pupil=clear_aberration=0_emccd=0'
        dark_file        = None

        z = zelda.Sensor('ROMAN-CGI')
        clear_pupil, zelda_pupil, center = z.read_files(path_raw, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=True, center=(175.5, 175.5), shift_method='interp')

        wave = zwfs.bandpass_values[bandpass]['wave']
        zelda_pupil_pos_noiseless = np.copy(zelda_pupil)
        opd_pos_noiseless = z.analyze(clear_pupil, zelda_pupil, wave=wave)

        # nagative grid
        zelda_pupil_file = f'pupil=zwfs_aberration=+1_emccd=1'
        clear_pupil_file = f'pupil=clear_aberration=0_emccd=1'
        dark_file        = None

        z = zelda.Sensor('ROMAN-CGI')
        clear_pupil, zelda_pupil, center = z.read_files(path_raw, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=True, center=(175.5, 175.5), shift_method='interp')

        wave = zwfs.bandpass_values[bandpass]['wave']
        zelda_pupil_pos = np.copy(zelda_pupil)
        opd_pos = z.analyze(clear_pupil, zelda_pupil, wave=wave)

        #%% additional post-processing

        # erode pupil to avoid edge effects
        pupil = (opd_neg_noiseless != 0)
        pupil_analysis = ndimage.binary_erosion(pupil, iterations=4)

        # subtract piston and save
        opd_neg_noiseless[pupil_analysis == 0] = np.nan
        opd_neg_noiseless = opd_neg_noiseless - np.nanmean(opd_neg_noiseless[pupil_analysis])
        fits.writeto(path_processed / 'opd_map_neg_noiseless.fits', opd_neg_noiseless, overwrite=True)

        opd_neg[pupil_analysis == 0] = np.nan
        opd_neg = opd_neg - np.nanmean(opd_neg[pupil_analysis])
        fits.writeto(path_processed / 'opd_map_neg.fits', opd_neg, overwrite=True)

        opd_pos_noiseless[pupil_analysis == 0] = np.nan
        opd_pos_noiseless = opd_pos_noiseless - np.nanmean(opd_pos_noiseless[pupil_analysis])
        fits.writeto(path_processed / 'opd_map_pos_noiseless.fits', opd_pos_noiseless, overwrite=True)

        opd_pos[pupil_analysis == 0] = np.nan
        opd_pos = opd_pos - np.nanmean(opd_pos[pupil_analysis])
        fits.writeto(path_processed / 'opd_map_pos.fits', opd_pos, overwrite=True)

        #%%
        opd_diff_noiseless = opd_pos_noiseless - opd_neg_noiseless

        opd_diff = opd_pos - opd_neg
        # opd_diff = imutils.sigma_filter(opd_diff, box=12, nsigma=3, iterate=True)

        data = np.stack((opd_diff_noiseless, opd_diff))
        fits.writeto(path_processed / 'differential_opd_maps.fits', data, overwrite=True)

        # plot
        plt.close('all')
        fig = plt.figure('Reconstruction error', figsize=(9, 7))
        fig.clf()

        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.1])

        cmap = mpl.cm.bwr
        norm = colors.Normalize(vmin=-2, vmax=2)

        ax = fig.add_subplot(gs[0])
        ax.set_rasterization_zorder(-1_000)

        cim = ax.imshow(opd_diff_noiseless, cmap=cmap, norm=norm)

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        # ax.set_title(f'bandpass={bandpass}, DM={dm_case}, Z$_\\mathrm{{index}}$={Zidx}, Z$_\\mathrm{{ampl}}$={Zamp*1000:.0f}pm')

        ax = fig.add_subplot(gs[1])
        cbar = fig.colorbar(cim, cax=ax)
        cbar.set_label('Reconstruction error [nm]')

        fig.subplots_adjust(left=0.02, right=0.84, bottom=0.03, top=0.94, wspace=0.1)

        # fig.savefig(path_processed / f'differential_opd_map.png', dpi=300)
