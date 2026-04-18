import numpy as np
import roman_preflight_proper
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import pyzelda.zelda as zelda
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

def build_grid(grid_size, grid_step):
    grid = np.arange(0, grid_size+1, grid_step)
    xgrid, ygrid = np.meshgrid(grid, grid)

    return grid, xgrid, ygrid


if __name__ == '__main__':
    __spec__ = None

    # Copy the default prescription file
    roman_preflight_proper.copy_here()

    root = Path('/Users/avigan/data/Roman/ZWFS/')

    vmag   = 2
    sptype = 'G0V'

    bandpass = '1B'
    dm_case  = '5e-9'   # flat, 3e-8, 5e-9, 2e-9

    grid_size = 30    # mas
    grid_step = 2     # mas

    generate = False
    analyze  = True

    # generate grid of offsets
    grid, xgrid, ygrid = build_grid(grid_size, grid_step)

    path = root / 'jitter_grid' / f'dm={dm_case}_bandpass={bandpass}'
    path_raw = path / 'raw'
    path_processed = path / 'processed'

    if generate:
        ref_star_properties = {
            'Vmag': vmag, 'spectral_type': sptype, 'magtype': 'vegamag',
            }

        # generate clear pupil image
        scene = zwfs.get_noiseless_zwfs_data(ref_star_properties, zwfs.PupilType.CLEAR, bandpass=bandpass, dm_case=dm_case)

        outpath = path_raw / 'jitter_offset_pupil=clear.fits'
        outputs.save_hdu_to_fits(scene, outdir=outpath.parent, filename=outpath.name, write_as_L1=False, overwrite=True)

        # generate ZWFS images
        for x_off_mas, y_off_mas in tqdm(zip(xgrid.flatten(), ygrid.flatten()), total=xgrid.size):
            optics_keywords = {
                'source_x_offset_mas': x_off_mas,
                'source_y_offset_mas': y_off_mas
                }

            scene = zwfs.get_noiseless_zwfs_data(ref_star_properties, zwfs.PupilType.ZWFS, bandpass=bandpass, dm_case=dm_case, optics_keywords=optics_keywords)

            outpath = path_raw / f'jitter_offset_pupil=zwfs_x={x_off_mas:03d}_y={y_off_mas:03d}.fits'
            outputs.save_hdu_to_fits(scene, outdir=outpath.parent, filename=outpath.name, write_as_L1=False, overwrite=True)

    if analyze:
        opd_map_size = 296
        pupil_size   = 292
        opd_maps = np.zeros((opd_map_size, opd_map_size, len(grid), len(grid)))

        for x in range(len(grid)):
            for y in range(len(grid)):
                x_off_mas = xgrid[y, x]
                y_off_mas = ygrid[y, x]

                zelda_pupil_file = f'jitter_offset_pupil=zwfs_x={x_off_mas:03d}_y={y_off_mas:03d}'
                clear_pupil_file = 'jitter_offset_pupil=clear'
                dark_file        = None

                z = zelda.Sensor('ROMAN-CGI')
                clear_pupil, zelda_pupil, center = z.read_files(path_raw, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=False, center=(175.5, 175.5))

                wave = bandpass_values[bandpass]['wave']
                opd = z.analyze(clear_pupil, zelda_pupil, wave=wave)

                opd_maps[:, :, y, x] = opd

        # erode pupil to avoid edge effects
        pupil = (opd_maps[..., 0, 0] != 0)
        pupil_analysis = ndimage.binary_erosion(pupil, iterations=4)

        for x in range(len(grid)):
            for y in range(len(grid)):
                opd = opd_maps[..., y, x]
                opd[pupil_analysis == 0] = np.nan
                opd = opd - np.nanmean(opd[pupil_analysis])
                opd_maps[..., y, x] = opd

        # save
        path_processed.mkdir(parents=True, exist_ok=True)
        hdu_list = []
        hdu = fits.PrimaryHDU()
        hdu.header['GRIDSIZE'] = grid_size
        hdu.header['GRIDSTEP'] = grid_step
        hdu_list.append(hdu)
        hdu = fits.ImageHDU(opd_maps.T, name='OPD_MAPS')
        hdu_list.append(hdu)
        hdu = fits.ImageHDU(xgrid, name='XGRID')
        hdu_list.append(hdu)
        hdu = fits.ImageHDU(ygrid, name='YGRID')
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
        for x in range(len(grid)):
            for y in range(len(grid)):
                opd = opd_maps[..., y, x]

                # projection on Zernike basis
                data = np.reshape(opd*pupil, (1, -1))
                data[:, mask == 0] = 0
                zcoeff = (basis @ data.T).squeeze() / mask.sum()

                # tip, tilt
                tip_nm_rms   = zcoeff[1]
                tilt_nm_rms  = zcoeff[2]

                opd_no_tt = opd - tip_nm_rms * zern_basis[1] - tilt_nm_rms * zern_basis[2]
                opd_maps_no_tt[..., y, x] = opd_no_tt

        opd_maps = opd_maps_no_tt

        # save
        path_processed.mkdir(parents=True, exist_ok=True)
        hdu_list = []
        hdu = fits.PrimaryHDU()
        hdu.header['GRIDSIZE'] = grid_size
        hdu.header['GRIDSTEP'] = grid_step
        hdu_list.append(hdu)
        hdu = fits.ImageHDU(opd_maps_no_tt.T, name='OPD_MAPS_NO_TIP_TILT')
        hdu_list.append(hdu)
        hdu = fits.ImageHDU(xgrid, name='XGRID')
        hdu_list.append(hdu)
        hdu = fits.ImageHDU(ygrid, name='YGRID')
        hdu_list.append(hdu)
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(path_processed / 'opd_maps_error.fits', overwrite=True)

        #%% compute error map

        # compute reconstruction error
        ref = np.where(grid == 0)[0][0]
        reference = opd_maps[..., ref, ref]
        opd_maps_error = opd_maps - reference[..., np.newaxis, np.newaxis]

        # build error map
        error_map = np.zeros((len(grid), len(grid)))
        for x in range(len(grid)):
            for y in range(len(grid)):
                error_map[y, x] = np.nanstd(opd_maps_error[..., y, x])

        #%% plot
        fig = plt.figure('Error map', figsize=(9, 7))
        fig.clf()

        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.1])

        cmap = mpl.cm.plasma
        norm = colors.Normalize(vmin=0, vmax=20)

        ax = fig.add_subplot(gs[0])
        cim = ax.pcolormesh(xgrid, ygrid, error_map, cmap=cmap, norm=norm)

        ax.set_xlabel('x offset [mas]')
        ax.set_ylabel('y offset [mas]')
        ax.set_title(f'bandpass={bandpass}, DM={dm_case}')

        ax.set_aspect('equal')

        ax = fig.add_subplot(gs[1])
        cbar = fig.colorbar(cim, cax=ax)
        cbar.locator = ticker.MultipleLocator(5)
        cbar.set_label('OPD error [nm rms]')

        fig.subplots_adjust(left=0.13, right=0.87, bottom=0.1, top=0.95, wspace=0.1)

        fig.savefig(path_processed / 'error_map.pdf', dpi=300)



