import numpy as np
import roman_preflight_proper
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import pyzelda.zelda as zelda

from pathlib import Path
from tqdm import tqdm

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

if __name__ == '__main__':
    __spec__ = None

    # Copy the default prescription file
    roman_preflight_proper.copy_here()

    root = Path('/Users/avigan/data/Roman/ZWFS/')

    vmag   = 2
    sptype = 'G0V'

    bandpass = '1B'
    dm_case  = 'flat'   # flat, 3e-8, 5e-9, 2e-9

    grid_size = 20    # mas
    grid_step = 5     # mas

    generate = True
    analyze  = False

    # generate grid of offsets
    grid = np.arange(-grid_size//2, grid_size//2+1, grid_step)
    xgrid, ygrid = np.meshgrid(grid, grid)

    path = root / 'jitter_grid' / f'dm={dm_case}'

    if generate:
        ref_star_properties = {
            'Vmag': vmag, 'spectral_type': sptype, 'magtype': 'vegamag',
            }

        # generate clear pupil image
        scene = zwfs.get_noiseless_zwfs_data(ref_star_properties, zwfs.PupilType.CLEAR, bandpass=bandpass, dm_case=dm_case)

        outpath = path / 'raw' / 'jitter_offset_pupil=clear.fits'
        outputs.save_hdu_to_fits(scene, outdir=outpath.parent, filename=outpath.name, write_as_L1=False, overwrite=True)

        # generate ZWFS images
        for x_off_mas, y_off_mas in tqdm(zip(xgrid.flatten(), ygrid.flatten()), total=xgrid.size):
            optics_keywords = {
                'source_x_offset_mas': x_off_mas,
                'source_y_offset_mas': y_off_mas
                }

            scene = zwfs.get_noiseless_zwfs_data(ref_star_properties, zwfs.PupilType.ZWFS, bandpass=bandpass, dm_case=dm_case, optics_keywords=optics_keywords)

            outpath = path / 'raw' / f'jitter_offset_pupil=zwfs_x={x_off_mas:+04d}_y={y_off_mas:+04d}.fits'
            outputs.save_hdu_to_fits(scene, outdir=outpath.parent, filename=outpath.name, write_as_L1=False, overwrite=True)

    if analyze:
        # ZELDA analysis
        files = sorted(list(path / 'raw').glob('jitter_offset_pupil=zwfs_*.fits'))
        zelda_pupil_files = [file.stem for file in files]
        clear_pupil_files = ['jitter_offset_pupil=clear']
        dark_file = None

        z = zelda.Sensor('ROMAN-CGI')

        clear_pupil, zelda_pupil, center = z.read_files(path, clear_pupil_files, zelda_pupil_files, dark_file, collapse_clear=True, collapse_zelda=False, center=(175.5, 175.5))

        wave = bandpass_values[bandpass]['wave']
        opd_maps = z.analyze(clear_pupil, zelda_pupil, wave=wave, ratio_limit=5)




