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
import corgisim

from scipy import interpolate
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm
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

    vmag     = 2
    sptype   = 'G0V'

    bandpass = '1F'
    dm_case  = 'flat'   # flat, 3e-8, 5e-9, 2e-9

    emgain   = 2
    exptime  = 1

    sampling = 200
    nexp     = 200

    # sampling = 1000
    # nexp     = 200

    sampling = 1000
    nexp     = 800

    generate = False
    observe  = False
    analyze  = False

    # paths
    path = root / 'time_series' / f'dm={dm_case}_bandpass={bandpass}'
    path_raw = path / 'raw'
    path_processed = path / 'processed'

    path_raw.mkdir(parents=True, exist_ok=True)
    path_processed.mkdir(parents=True, exist_ok=True)

    # read OS11 sequence
    batch_info = fits.getdata(Path(corgisim.__file__).parent / 'data' / 'hlc_os11_inputs.fits').transpose()
    batch_time = (batch_info[0] - 170) * 3600
    batch_id   = batch_info[2]
    batch_roll = batch_info[3]
    batch_zern = batch_info[4:46]

    t_frames = batch_time

    noll_index = np.arange(4, 46, 1)
    z_ampl     = batch_zern.transpose()

    # time frames with no data
    seq_change_id = np.where(np.diff(batch_id) != 0)[0]
    exclusion_times = []
    for cid in seq_change_id:
        t = (batch_time[cid], batch_time[cid+1])
        exclusion_times.append(t)

    # t_frames = np.arange(0, batch_time.max(), sampling)
    # n_frames = len(t_frames)

    # noll_index = np.arange(4, 46, 1)
    # z_ampl     = np.zeros((n_frames, len(noll_index)))
    # for zidx in range(len(noll_index)):
    #     f_interp = interpolate.interp1d(batch_time, batch_zern[zidx])
    #     z_ampl[:, zidx] = f_interp(t_frames)

    plt.close('all')

    if generate:
        # simulate data
        ref_star_properties = {
            'Vmag': vmag, 'spectral_type': sptype, 'magtype': 'vegamag',
            }

        # noiseless clear pupil data
        pupil_type = zwfs.PupilType.CLEAR

        zernike_keywords = {
            'zindex': noll_index,
            'zval_m': z_ampl[0],
            }

        noiseless_clear = zwfs.generate_zwfs_data(ref_star_properties, pupil_type, bandpass=bandpass, dm_case=dm_case, optics_keywords=zernike_keywords)

        outpath = path_raw / f'timeindex={0:04d}_pupil={pupil_type.value}_emccd=0.fits'
        fits.writeto(outpath, noiseless_clear.host_star_image.data, overwrite=True)
        with fits.open(outpath, mode='update') as hdul:
            hdul[0].header['TIME'] = (0.0, '[s] Time since start of OS11 sequence')
            for idx, znoll in enumerate(noll_index):
                hdul[0].header[f'Z{znoll}'] = (z_ampl[0, idx]*1e12, f'[pm] Z{znoll} amplitude')

        # noiseless ZWFS pupil data
        import time as ttime
        pupil_type = zwfs.PupilType.ZWFS
        for tidx, time in tqdm(enumerate(t_frames), total=len(t_frames)):
            zernike_keywords = {
                'zindex': noll_index,
                'zval_m': z_ampl[tidx],
                }

            # noiseless reference data
            noiseless_zwfs = zwfs.generate_zwfs_data(ref_star_properties, pupil_type, bandpass=bandpass, dm_case=dm_case, optics_keywords=zernike_keywords)

            outpath = path_raw / f'timeindex={tidx:04d}_pupil={pupil_type.value}_emccd=0.fits'
            fits.writeto(outpath, noiseless_zwfs.host_star_image.data, overwrite=True)
            with fits.open(outpath, mode='update') as hdul:
                hdul[0].header['TIME'] = (time, '[s] Time since start of OS11 sequence')
                for idx, znoll in enumerate(noll_index):
                    hdul[0].header[f'Z{znoll}'] = (z_ampl[tidx, idx]*1e12, f'[pm] Z{znoll} amplitude')

    #%% sequence sampling

    # identify usable samples
    samples = np.arange(0, t_frames.max(), sampling)
    idx_obs    = np.array([np.argmin(np.abs(t-t_frames)) for t in samples])

    diff = samples - t_frames[idx_obs]
    idx_obs = idx_obs[np.abs(diff) < 70]

    t_index_obs = idx_obs
    t_obs       = t_frames[idx_obs]
    z_ampl_obs  = z_ampl[idx_obs]

    t_oversample = np.arange(0, t_frames.max(), 1)
    z_ampl_oversample = np.zeros((len(t_oversample), 42))
    for zidx in range(42):
        znoll = noll_index[zidx]
        z_fun = interpolate.interp1d(t_frames, z_ampl[:, zidx])
        z_ampl_oversample[:, zidx] = z_fun(t_oversample)

    for t in exclusion_times:
        t0, t1 = t
        z_ampl_oversample[(t0 <= t_oversample) & (t_oversample <= t1), :] = np.nan

    # plot
    fig = plt.figure('OS11 sequence sampling', figsize=(10, 7))
    fig.clf()

    ax = fig.add_subplot()
    ax.plot(t_oversample/3600, np.sqrt(np.sum(z_ampl_oversample**2, axis=1)) * 1e12)
    # ax.plot(t_obs/3600, np.sqrt(np.sum(z_ampl_obs**2, axis=1)) * 1e12, marker='o', ms=5, linestyle='none')

    ax.set_xlabel('OS11 sequence time [h]')
    ax.set_xlim(0, 20)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.set_ylabel('Wavefront variations [pm]')

    ax.set_title('Z4 - Z45')

    fig.subplots_adjust(left=0.1, right=0.96, bottom=0.11, top=0.95)

    fig.savefig(path_processed / f'os11_sequence_sampling={sampling}.png')

    #%%
    if observe:
        pupil_img_size = 351

        if sampling < 60:
            raise ValueError('Sampling cannot be smaller than 60 sec')

        if (exptime*nexp) > sampling:
            raise ValueError('Sampling and integration times are incompatible')

        #%% generate NOISELESS clear pupil and ZWFS images

        # clear pupil data
        pupil_type = zwfs.PupilType.CLEAR

        inpath = path_raw / f'timeindex={0:04d}_pupil={pupil_type.value}_emccd=0.fits'
        noiseless_clear, hdr = fits.getdata(inpath, header=True)

        outpath = path_processed / f'sampling={sampling}_pupil={pupil_type.value}_emccd=0_ref.fits'
        fits.writeto(outpath, noiseless_clear, hdr, overwrite=True)
        with fits.open(outpath, mode='update') as hdul:
            hdul[0].header['SAMPLING']  = (sampling, '[s] Temporal sampling of OS11 sequence')

        # ZWFS pupil data
        pupil_type = zwfs.PupilType.ZWFS
        exposures_zwfs = np.zeros((len(t_obs), pupil_img_size, pupil_img_size))
        for idx, tidx in enumerate(t_index_obs):
            try:
                inpath = path_raw / f'timeindex={tidx:04d}_pupil={pupil_type.value}_emccd=0.fits'
                noiseless_zwfs, hdr = fits.getdata(inpath, header=True)

                exposures_zwfs[idx] = noiseless_zwfs
            except:
                pass

        outpath = path_processed / f'sampling={sampling}_pupil={pupil_type.value}_emccd=0_ref.fits'
        fits.writeto(outpath, exposures_zwfs, hdr, overwrite=True)
        with fits.open(outpath, mode='update') as hdul:
            hdul[0].header['SAMPLING']  = (sampling, '[s] Temporal sampling of OS11 sequence')

        #%% generate EMCCD-observed clear pupil and ZWFS images

        # clear pupil data
        pupil_type = zwfs.PupilType.CLEAR

        inpath = path_raw / f'timeindex={0:04d}_pupil={pupil_type.value}_emccd=0.fits'
        noiseless_clear, hdr = fits.getdata(inpath, header=True)

        scene = zwfs.DummyScene(noiseless_clear)
        exposures = zwfs.observe_with_emccd(scene, emgain=emgain, photon_counting=False, exposure_time=exptime, num_exposures=nexp)
        data = np.array([exp.data.astype(float) for exp in exposures])
        exposure_clear = np.mean(data, axis=0)

        outpath = path_processed / f'sampling={sampling}_pupil={pupil_type.value}_emccd=1_texp={exptime*nexp:.0f}.fits'
        fits.writeto(outpath, exposure_clear, hdr, overwrite=True)
        with fits.open(outpath, mode='update') as hdul:
            hdul[0].header['SAMPLING']  = (sampling, '[s] Temporal sampling of OS11 sequence')
            hdul[0].header['EMGAIN']  = (emgain, 'EMCCD gain')
            hdul[0].header['EXPTIME'] = (exptime, '[s] Integration time of individual exposures')
            hdul[0].header['NEXP']    = (nexp, 'Number of exposures')

        # ZWFS pupil data
        pupil_type = zwfs.PupilType.ZWFS
        exposures_zwfs = np.zeros((len(t_obs), pupil_img_size, pupil_img_size))
        for idx, tidx in enumerate(t_index_obs):
            try:
                inpath = path_raw / f'timeindex={tidx:04d}_pupil={pupil_type.value}_emccd=0.fits'
                noiseless_zwfs, hdr = fits.getdata(inpath, header=True)

                scene = zwfs.DummyScene(noiseless_zwfs)
                exposures = zwfs.observe_with_emccd(scene, emgain=emgain, photon_counting=False, exposure_time=exptime, num_exposures=nexp)
                data = np.array([exp.data.astype(float) for exp in exposures])
                exposure_zwfs = np.mean(data, axis=0)

                exposures_zwfs[idx] = exposure_zwfs
            except:
                pass

        outpath = path_processed / f'sampling={sampling}_pupil={pupil_type.value}_emccd=1_texp={exptime*nexp:.0f}.fits'
        fits.writeto(outpath, exposures_zwfs, hdr, overwrite=True)
        with fits.open(outpath, mode='update') as hdul:
            hdul[0].header['SAMPLING']  = (sampling, '[s] Temporal sampling of OS11 sequence')
            hdul[0].header['EMGAIN']  = (emgain, 'EMCCD gain')
            hdul[0].header['EXPTIME'] = (exptime, '[s] Integration time of individual exposures')
            hdul[0].header['NEXP']    = (nexp, 'Number of exposures')

    #%%
    if analyze:
        plt.close('all')

        opd_map_size = 296
        pupil_size   = 292

        # reference case - NOISELESS
        zelda_pupil_file = f'sampling={sampling}_pupil=zwfs_emccd=0_ref'
        clear_pupil_file = f'sampling={sampling}_pupil=clear_emccd=0_ref'
        dark_file        = None

        z = zelda.Sensor('ROMAN-CGI')
        clear_pupil, zelda_pupil, center = z.read_files(path_processed, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=False, center=(175.5, 175.5), shift_method='interp')

        wave = zwfs.bandpass_values[bandpass]['wave']
        opd_maps_ref = z.analyze(clear_pupil, zelda_pupil, wave=wave)

        opd_maps_ref -= opd_maps_ref[0]

        outpath = path_processed / f'sampling={sampling}_emccd=0_opd_ref.fits'
        fits.writeto(outpath, opd_maps_ref, overwrite=True)

        # observations
        zelda_pupil_file = f'sampling={sampling}_pupil=zwfs_emccd=1_texp={exptime*nexp:.0f}'
        clear_pupil_file = f'sampling={sampling}_pupil=clear_emccd=1_texp={exptime*nexp:.0f}'
        dark_file        = None

        z = zelda.Sensor('ROMAN-CGI')
        clear_pupil, zelda_pupil, center = z.read_files(path_processed, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=False, center=(175.5, 175.5), shift_method='interp')

        wave = zwfs.bandpass_values[bandpass]['wave']
        opd_maps = z.analyze(clear_pupil, zelda_pupil, wave=wave)

        opd_maps -= opd_maps[0]

        outpath = path_processed / f'sampling={sampling}_emccd=1_texp={exptime*nexp:.0f}_opd.fits'
        fits.writeto(outpath, opd_maps, overwrite=True)

        #%%
        psd_cutoff = 100  # cycles/pupil

        # erode pupil to avoid edge effects
        pupil = (opd_maps_ref[1] != 0)
        # pupil_analysis = ndimage.binary_erosion(pupil, iterations=4)
        pupil_analysis = pupil

        # subtract piston
        # opd_maps_ref[pupil_analysis == 0] = np.nan
        # opd_maps_ref = opd_maps_ref - np.nanmean(opd_maps_ref[pupil_analysis])

        # opd_maps[pupil_analysis == 0] = np.nan
        # opd_maps = opd_maps - np.nanmean(opd_maps[pupil_analysis])

        # compute PSD
        psd_cube_ref = zseq.compute_psd(None, np.nan_to_num(opd_maps_ref), freq_cutoff=psd_cutoff, return_fft=False, pupil_mask=pupil_analysis)
        psd_int_ref, psd_bnds_noiseless = zseq.integrate_psd(None, psd_cube_ref, freq_cutoff=psd_cutoff)

        psd_cube = zseq.compute_psd(None, np.nan_to_num(opd_maps), freq_cutoff=psd_cutoff, return_fft=False, pupil_mask=pupil_analysis)
        psd_int, psd_bnds = zseq.integrate_psd(None, psd_cube, freq_cutoff=psd_cutoff)

        #%% plot

        # PSD - 1D
        fig = plt.figure('PSD - 1D', figsize=(12, 7))
        fig.clf()

        ax = fig.add_subplot(111)

        # for icycle, cycles in enumerate((1, 3, 10, 30, 60)):
        for icycle, cycles in enumerate((1, 3, 5, 12, 20)):
            ax.plot(t_obs/3600, psd_int_ref[cycles], color=f'C{icycle}', linestyle=':', zorder=-1000)
            ax.plot(t_obs/3600, psd_int[cycles], color=f'C{icycle}', linestyle='-', label=f'{cycles} c/p', zorder=-1000)

        for t in exclusion_times:
            t0, t1 = t
            ax.axvspan(t0/3600, t1/3600, color='w')

        ax.set_xlabel('OS11 sequence time [h]')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.set_xlim(0, 20)

        ax.set_yscale('log')
        ax.set_ylabel('PSD [nm rms / (c/p)]')
        ax.set_ylim(5e-5, 5e-2)

        ax.set_title(f'bandpass={bandpass}, DM={dm_case}, t$_{{\\mathrm{{exp}}}}$={nexp*exptime:.0f}s')

        ax.legend(loc='lower right')

        fig.subplots_adjust(left=0.1, right=0.96, bottom=0.11, top=0.94)

        fig.savefig(path_processed / f'os11_reconstruction_error_psd_DM={dm_case}_sampling={sampling}_texp={exptime*nexp:.0f}_1d.png', dpi=300)

        #%%
        # PSD - 2D
        fig = plt.figure('PSD - 2D', figsize=(17, 7))
        fig.clf()

        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.1])

        cmap = mpl.cm.plasma
        norm = colors.LogNorm(vmin=1e-3, vmax=0.1)

        ax = fig.add_subplot(gs[0])
        ax.set_rasterization_zorder(-1_000)

        freq = np.arange(psd_cutoff)
        psd_int_plot = psd_int_ref.copy()
        psd_int_plot[psd_int_plot == 0] = 1e-10
        cim = ax.pcolormesh(t_obs/3600, freq, psd_int_plot, cmap=cmap, norm=norm, zorder=-10_000)
        # ax.contour(freq, xgrid, psd_int_plot.T, levels=[0.03, 0.1, 0.3, 1.0], colors=['w', 'w', 'k', 'k'])
        # ax.axhline(4, linestyle=':', color='w', label='Jitter (no LOWFS)')
        # ax.text(0.3, 4.3, 'Jitter (no LOWFS)', fontsize='x-small', color='w')

        for t in exclusion_times:
            t0, t1 = t
            ax.axvspan(t0/3600, t1/3600, color='w')

        ax.set_xlabel('OS11 sequence time [h]')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.set_xlim(0, 20)

        ax.set_ylabel('Spatial frequency [c/p]')
        ax.set_ylim(0, 40)

        ax.set_title(f'bandpass={bandpass}, DM={dm_case}, noiseless')

        ax = fig.add_subplot(gs[1], sharex=ax, sharey=ax)
        ax.set_rasterization_zorder(-1_000)

        freq = np.arange(psd_cutoff)
        psd_int_plot = psd_int.copy()
        psd_int_plot[psd_int_plot == 0] = 1e-10
        cim = ax.pcolormesh(t_obs/3600, freq, psd_int_plot, cmap=cmap, norm=norm, zorder=-10_000)
        # ax.contour(freq, xgrid, psd_int_plot.T, levels=[0.03, 0.1, 0.3, 1.0], colors=['w', 'w', 'k', 'k'])
        # ax.axhline(4, linestyle=':', color='w', label='Jitter (no LOWFS)')
        # ax.text(0.3, 4.3, 'Jitter (no LOWFS)', fontsize='x-small', color='w')

        for t in exclusion_times:
            t0, t1 = t
            ax.axvspan(t0/3600, t1/3600, color='w')

        ax.set_xlabel('OS11 sequence time [h]')

        ax.set_title(f'bandpass={bandpass}, DM={dm_case}, t$_{{\\mathrm{{exp}}}}$={nexp*exptime:.0f}s')

        ax = fig.add_subplot(gs[2])
        cbar = fig.colorbar(cim, cax=ax)
        cbar.set_label('PSD [nm rms / (c/p)]')

        fig.subplots_adjust(left=0.06, right=0.92, bottom=0.11, top=0.94, wspace=0.17)

        fig.savefig(path_processed / f'os11_reconstruction_error_psd_DM={dm_case}_sampling={sampling}_texp={exptime*nexp:.0f}_2d.png', dpi=300)
