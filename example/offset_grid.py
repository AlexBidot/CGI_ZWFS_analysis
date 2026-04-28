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

from corgisim import outputs, scene, instrument

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

def build_grid(grid_size, grid_step, grid_geom):
    grid = np.arange(0, grid_size+1, grid_step)

    if grid_geom == '1d':
        xgrid = grid
        ygrid = np.zeros_like(grid)
    elif grid_geom == '2d':
        xgrid, ygrid = np.meshgrid(grid, grid)

    return grid, xgrid.flatten(), ygrid.flatten()


if __name__ == '__main__':
    __spec__ = None

    # Copy the default prescription file
    roman_preflight_proper.copy_here()

    root = Path('/Users/avigan/data/Roman/ZWFS/')

    vmag   = 2
    sptype = 'G0V'

    bandpass = '1B'
    dm_case  = '5e-9'   # flat, 3e-8, 5e-9, 2e-9

    grid_geom = '1d'  # 1d, 2d
    grid_size = 20    # mas
    grid_step = 0.1   # mas

    generate = False
    analyze  = True

    # generate grid of offsets
    grid, xgrid, ygrid = build_grid(grid_size, grid_step, grid_geom)

    path = root / 'offset_grid' / f'dm={dm_case}_bandpass={bandpass}'
    path_raw = path / 'raw'
    path_processed = path / 'processed'

    if generate:
        ref_star_properties = {
            'Vmag': vmag, 'spectral_type': sptype, 'magtype': 'vegamag',
            }

        # generate clear pupil image
        scene = zwfs.get_zwfs_data(ref_star_properties, zwfs.PupilType.CLEAR, bandpass=bandpass, dm_case=dm_case)

        outpath = path_raw / 'offset_pupil=clear.fits'
        outputs.save_hdu_to_fits(scene, outdir=outpath.parent, filename=outpath.name, write_as_L1=False, overwrite=True)

        # generate ZWFS images
        for x_off_mas, y_off_mas in tqdm(zip(xgrid, ygrid), total=xgrid.size):
            optics_keywords = {
                'source_x_offset_mas': x_off_mas,
                'source_y_offset_mas': y_off_mas
                }

            scene = zwfs.get_zwfs_data(ref_star_properties, zwfs.PupilType.ZWFS, bandpass=bandpass, dm_case=dm_case, optics_keywords=optics_keywords)

            outpath = path_raw / f'offset_pupil=zwfs_x={x_off_mas:04.1f}_y={y_off_mas:04.1f}.fits'
            outputs.save_hdu_to_fits(scene, outdir=outpath.parent, filename=outpath.name, write_as_L1=False, overwrite=True)

    if analyze:
        opd_map_size = 296
        pupil_size   = 292

        opd_maps = np.zeros((xgrid.size, opd_map_size, opd_map_size))
        for idx, (x_off_mas, y_off_mas) in enumerate(zip(xgrid, ygrid)):
            zelda_pupil_file = f'offset_pupil=zwfs_x={x_off_mas:04.1f}_y={y_off_mas:04.1f}'
            clear_pupil_file = 'offset_pupil=clear'
            dark_file        = None

            z = zelda.Sensor('ROMAN-CGI')
            clear_pupil, zelda_pupil, center = z.read_files(path_raw, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=False, center=(175.5, 175.5), shift_method='interp')

            wave = bandpass_values[bandpass]['wave']
            opd = z.analyze(clear_pupil, zelda_pupil, wave=wave)

            opd_maps[idx, :, :] = opd

        # erode pupil to avoid edge effects
        pupil = (opd_maps[0] != 0)
        pupil_analysis = ndimage.binary_erosion(pupil, iterations=4)

        for iopd, opd in enumerate(opd_maps):
            opd[pupil_analysis == 0] = np.nan
            opd = opd - np.nanmean(opd[pupil_analysis])
            opd_maps[idx] = opd

        # save
        path_processed.mkdir(parents=True, exist_ok=True)
        hdu_list = []
        hdu = fits.PrimaryHDU()
        hdu.header['GRIDGEOM'] = grid_geom
        hdu.header['GRIDSIZE'] = grid_size
        hdu.header['GRIDSTEP'] = grid_step
        hdu_list.append(hdu)
        hdu = fits.ImageHDU(opd_maps, name='OPD_MAPS')
        hdu_list.append(hdu)
        hdu = fits.BinTableHDU.from_columns([fits.Column(name='XGRID', format='D', array=xgrid)], name='XGRID')
        hdu_list.append(hdu)
        hdu = fits.BinTableHDU.from_columns([fits.Column(name='YGRID', format='D', array=ygrid)], name='YGRID')
        hdu_list.append(hdu)
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(path_processed / 'opd_maps.fits', overwrite=True)

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

        opd_maps_no_tt = np.zeros_like(opd_maps)
        for iopd, opd in enumerate(opd_maps):
            # projection on Zernike basis
            data = np.reshape(opd*pupil, (1, -1))
            data[:, mask == 0] = 0
            zcoeff = (basis @ data.T).squeeze() / mask.sum()

            # tip, tilt
            tip_nm_rms  = zcoeff[1]
            tilt_nm_rms = zcoeff[2]

            opd_no_tt = opd - tip_nm_rms * zern_basis[1] - tilt_nm_rms * zern_basis[2]
            opd_maps_no_tt[iopd] = opd_no_tt

        opd_maps = opd_maps_no_tt

        # save
        path_processed.mkdir(parents=True, exist_ok=True)
        hdu_list = []
        hdu = fits.PrimaryHDU()
        hdu.header['GRIDSIZE'] = grid_size
        hdu.header['GRIDSTEP'] = grid_step
        hdu_list.append(hdu)
        hdu = fits.ImageHDU(opd_maps_no_tt, name='OPD_MAPS_NO_TIP_TILT')
        hdu_list.append(hdu)
        hdu = fits.BinTableHDU.from_columns([fits.Column(name='XGRID', format='D', array=xgrid)], name='XGRID')
        hdu_list.append(hdu)
        hdu = fits.BinTableHDU.from_columns([fits.Column(name='YGRID', format='D', array=ygrid)], name='YGRID')
        hdu_list.append(hdu)
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(path_processed / 'opd_maps_no_tip_tilt.fits', overwrite=True)

        #%% compute error map

        # compute reconstruction error
        ref = np.where(grid == 0)[0][0]
        reference = opd_maps[ref]
        opd_maps_error = opd_maps - reference

        # build error map
        error_maps = np.zeros((xgrid.size, 3))
        for iopd, opd in enumerate(opd_maps_error):
            error_maps[iopd, 0] = np.nanpercentile(opd, 1)
            error_maps[iopd, 1] = np.nanpercentile(opd, 99)
            error_maps[iopd, 2] = np.nanstd(opd)

        # save
        path_processed.mkdir(parents=True, exist_ok=True)
        hdu_list = []
        hdu = fits.PrimaryHDU()
        hdu.header['GRIDSIZE'] = grid_size
        hdu.header['GRIDSTEP'] = grid_step
        hdu_list.append(hdu)
        hdu = fits.ImageHDU(opd_maps_error, name='RECONSTRUCTION_ERROR')
        hdu_list.append(hdu)
        hdu = fits.BinTableHDU.from_columns([fits.Column(name='XGRID', format='D', array=xgrid)], name='XGRID')
        hdu_list.append(hdu)
        hdu = fits.BinTableHDU.from_columns([fits.Column(name='YGRID', format='D', array=ygrid)], name='YGRID')
        hdu_list.append(hdu)
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(path_processed / 'reconstruction_error.fits', overwrite=True)

        #%% compute PSD of error maps
        psd_cutoff = 100  # cycles/pupil
        psd_cube = zseq.compute_psd(None, np.nan_to_num(opd_maps_error), freq_cutoff=psd_cutoff, return_fft=False, pupil_mask=pupil_analysis)
        psd_int, psd_bnds = zseq.integrate_psd(None, psd_cube, freq_cutoff=psd_cutoff)

        #%% plot
        if grid_geom == '1d':
            # OPD error
            fig = plt.figure('OPD reconstruction error', figsize=(9, 7))
            fig.clf()

            ax = fig.add_subplot()
            error_maps_plot = error_maps.copy()
            error_maps_plot[0] = np.nan
            ax.semilogy(xgrid, error_maps_plot[:, 2], linestyle='-', label='RMS')
            ax.semilogy(xgrid, error_maps_plot[:, 1]-error_maps[iopd, 0], linestyle='--', label='PtV')
            ax.axvline(4, linestyle=':', color='k', label='Jitter (no LOWFS)')

            ax.set_xlabel('Offset [mas]')
            ax.set_xlim(0, grid_size)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

            ax.set_ylabel('Reconstruction error [nm]')
            ax.set_ylim(bottom=0.1, top=100)

            ax.set_title(f'bandpass={bandpass}, DM={dm_case}')

            ax.legend()

            fig.subplots_adjust(left=0.13, right=0.87, bottom=0.1, top=0.95, wspace=0.1)

            fig.savefig(path_processed / f'reconstruction_error_DM={dm_case}.png', dpi=300)

            # PSD
            fig = plt.figure('PSD', figsize=(9, 7))
            fig.clf()

            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.1])

            cmap = mpl.cm.plasma
            norm = colors.LogNorm(vmin=0.01, vmax=5)

            ax = fig.add_subplot(gs[0])
            ax.set_rasterization_zorder(-1_000)

            freq = np.arange(psd_cutoff)
            psd_int_plot = psd_int.copy()
            psd_int_plot[psd_int_plot == 0] = 1e-10
            cim = ax.pcolormesh(freq, xgrid, psd_int_plot.T, cmap=cmap, norm=norm, zorder=-10_000)
            ax.contour(freq, xgrid, psd_int_plot.T, levels=[0.03, 0.1, 0.3, 1.0], colors=['w', 'w', 'k', 'k'])
            ax.axhline(4, linestyle=':', color='w', label='Jitter (no LOWFS)')
            ax.text(0.3, 4.3, 'Jitter (no LOWFS)', fontsize='x-small', color='w')

            ax.set_xlabel('Spatial frequency [c/p]')
            ax.set_xlim(0, 60)

            ax.set_ylabel('Offset [mas]')
            ax.set_ylim(0, 20)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

            ax.set_title(f'bandpass={bandpass}, DM={dm_case}')

            ax = fig.add_subplot(gs[1])
            cbar = fig.colorbar(cim, cax=ax)
            ax.axhline(0.03, color='w')
            ax.axhline(0.1, color='w')
            ax.axhline(0.3, color='k')
            ax.axhline(1.0, color='k')
            cbar.set_label('PSD [nm rms / (c/p)]')

            fig.subplots_adjust(left=0.1, right=0.87, bottom=0.11, top=0.94, wspace=0.1)

            fig.savefig(path_processed / f'reconstruction_error_psd_DM={dm_case}.png', dpi=300)


        # fig = plt.figure('Error map', figsize=(9, 7))
        # fig.clf()

        # gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.1])

        # cmap = mpl.cm.plasma
        # norm = colors.Normalize(vmin=0, vmax=20)

        # ax = fig.add_subplot(gs[0])
        # cim = ax.pcolormesh(xgrid, ygrid, error_map, cmap=cmap, norm=norm)

        # ax.set_xlabel('x offset [mas]')
        # ax.set_ylabel('y offset [mas]')
        # ax.set_title(f'bandpass={bandpass}, DM={dm_case}')

        # ax.set_aspect('equal')

        # ax = fig.add_subplot(gs[1])
        # cbar = fig.colorbar(cim, cax=ax)
        # cbar.locator = ticker.MultipleLocator(5)
        # cbar.set_label('OPD error [nm rms]')

        # fig.subplots_adjust(left=0.13, right=0.87, bottom=0.1, top=0.95, wspace=0.1)

        # fig.savefig(path_processed / 'error_map.pdf', dpi=300)
