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
    dm_case  = '5e-9'   # flat, 3e-8, 5e-9, 2e-9

    # paths
    path = root / 'basic_simulation' / f'dm={dm_case}_bandpass={bandpass}'
    path_raw = path / 'raw'
    path_processed = path / 'processed'

    path_raw.mkdir(parents=True, exist_ok=True)
    path_processed.mkdir(parents=True, exist_ok=True)

    #%% simulate data
    ref_star_properties = {
        'Vmag': vmag, 'spectral_type': sptype, 'magtype': 'vegamag',
        }

    # clear pupil image
    pupil_type = zwfs.PupilType.CLEAR
    clear_pupil = zwfs.generate_zwfs_data(ref_star_properties, pupil_type, bandpass=bandpass, dm_case=dm_case)
    outpath = path_raw / f'pupil={pupil_type.value}_emccd=0.fits'
    fits.writeto(outpath, clear_pupil.host_star_image.data, overwrite=True)
    clear_pupil = clear_pupil.host_star_image.data

    # ZWFS image
    pupil_type = zwfs.PupilType.ZWFS
    zwfs_pupil = zwfs.generate_zwfs_data(ref_star_properties, pupil_type, bandpass=bandpass, dm_case=dm_case)
    outpath = path_raw / f'pupil={pupil_type.value}_emccd=0.fits'
    fits.writeto(outpath, zwfs_pupil.host_star_image.data, overwrite=True)
    zwfs_pupil = zwfs_pupil.host_star_image.data

    #%%
    fig = plt.figure('ZWFS images', figsize=(14.5, 7))
    fig.clf()

    ax = fig.add_subplot(121)
    ax.imshow(clear_pupil, vmin=0, vmax=5e4, cmap='inferno')
    ax.set_title('Clear pupil')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    ax = fig.add_subplot(122, sharex=ax, sharey=ax)
    ax.imshow(zwfs_pupil, vmin=0, vmax=5e4, cmap='inferno')
    ax.set_title('ZWFS data')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.94, wspace=0.01)
    fig.savefig(path_processed / 'zwfs_images.pdf', dpi=300)

    #%%
    opd_map_size = 296
    pupil_size   = 292

    zelda_pupil_file = f'pupil=zwfs_emccd=0'
    clear_pupil_file = f'pupil=clear_emccd=0'
    dark_file        = None

    z = zelda.Sensor('ROMAN-CGI')
    clear_pupil, zelda_pupil, center = z.read_files(path_raw, [clear_pupil_file], [zelda_pupil_file], dark_file, collapse_clear=True, collapse_zelda=True, center=(175.5, 175.5), shift_method='interp')

    wave = zwfs.bandpass_values[bandpass]['wave']
    opd_ref = z.analyze(clear_pupil, zelda_pupil, wave=wave)

    # erode pupil to avoid edge effects
    pupil = (opd_ref != 0)
    pupil_analysis = ndimage.binary_erosion(pupil, iterations=4)

    # subtract piston and save
    opd_ref[pupil_analysis == 0] = np.nan
    opd_ref = opd_ref - np.nanmean(opd_ref[pupil_analysis])
    fits.writeto(path_processed / 'opd_map_ref.fits', opd_ref, overwrite=True)

    #%% plot
    plt.close('all')
    fig = plt.figure('OPD map', figsize=(9, 7))
    fig.clf()

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.1])

    cmap = mpl.cm.bwr
    norm = colors.Normalize(vmin=-150, vmax=150)

    ax = fig.add_subplot(gs[0])
    ax.set_rasterization_zorder(-1_000)

    cim = ax.imshow(opd_ref, cmap=cmap, norm=norm)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    #ax.set_title(f'bandpass={bandpass}, DM={dm_case}')
    ax.set_title(f'Reconstructed OPD map')

    ax = fig.add_subplot(gs[1])
    cbar = fig.colorbar(cim, cax=ax)
    cbar.set_label('Reconstruction error [nm]')

    fig.subplots_adjust(left=0.02, right=0.84, bottom=0.03, top=0.94, wspace=0.1)

    fig.savefig(path_processed / f'opd_map.pdf', dpi=300)
