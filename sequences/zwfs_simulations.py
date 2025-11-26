#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

from pathlib import Path
from astropy.io import fits

import cgisim as cgisim
import proper

import roman_preflight_proper
from roman_preflight_proper import trim

import pyzelda.zelda as zelda


if __name__ == '__main__':
    __spec__ = None

    path = Path('~/data/Roman/ZWFS/').expanduser()

    star_SpT  = 'a0v'
    star_Vmag = 2.0

    use_eccd     = False
    eccd_gain    = 1.0
    eccd_exptime = 1.0

    cgi_mode  = 'excam'
    cor_type  = 'zwfs'
    bandpass  = '1b'
    polaxis   = -10       # compute images for mean X+Y polarization (don't compute at each polarization)

    dm_shape = 'flat_wfe'
    # dm_shape = 'ni_2e-9'

    # DM shapes
    dm1v = proper.prop_fits_read(roman_preflight_proper.lib_dir+f'/examples/hlc_{dm_shape}_dm1_v.fits')
    dm2v = proper.prop_fits_read(roman_preflight_proper.lib_dir+f'/examples/hlc_{dm_shape}_dm2_v.fits')

    # additional errors
    zindex = None #[10, 11]
    zval   = None #[500e-9, 500e-9]

    # parameters
    params = {
        'use_errors':1, 'use_pupil_lens':1, 'use_fpm':1,
        'use_dm1':1, 'dm1_v': dm1v, 'use_dm2':1, 'dm2_v': dm2v,
        }
    if zindex is not None:
        params['zindex'] = zindex
        params['zval_m'] = zval

    if use_eccd:
        eccd_params = {'gain':eccd_gain, 'exptime':eccd_exptime}
    else:
        eccd_params = None

    # wavelength
    info_dir = cgisim.lib_dir + '/cgisim_info_dir/'
    mode_data, bandpass_data = cgisim.cgisim_read_mode(cgi_mode, cor_type, bandpass, info_dir)
    wave = bandpass_data['lam0_um']*1e-6

    # clean pupil image simulation
    params['use_fpm'] = 0
    output_clear = path / f'zwfs_SpT={star_SpT}_Vmag={star_Vmag}_dm={dm_shape}_clear.fits'
    pupil, counts = cgisim.rcgisim(cgi_mode, cor_type, bandpass, polaxis, params,
                                   star_spectrum=star_SpT, star_vmag=star_Vmag,
                                   ccd=eccd_params, output_file=str(output_clear)
                                   )

    # ZWFS image simulation
    params['use_fpm'] = 1
    output_zwfs = path / f'zwfs_SpT={star_SpT}_Vmag={star_Vmag}_dm={dm_shape}_zwfs.fits'
    pupil, counts = cgisim.rcgisim(cgi_mode, cor_type, bandpass, polaxis, params,
                                   star_spectrum=star_SpT, star_vmag=star_Vmag,
                                   ccd=eccd_params, output_file=str(output_zwfs)
                                   )

    # ZELDA analysis
    clear_pupil_files = [output_clear.stem]
    zelda_pupil_files = [output_zwfs.stem]
    dark_file = None

    z = zelda.Sensor('ROMAN-CGI')

    clear_pupil, zelda_pupil, center = z.read_files(path, clear_pupil_files, zelda_pupil_files, dark_file,
                                                    collapse_clear=True, collapse_zelda=False, center=(175.5, 175.5))

    opd_map = z.analyze(clear_pupil, zelda_pupil, wave=wave, ratio_limit=5)
    if opd_map.ndim == 3:
        opd_map = opd_map.mean(axis=0)

    output_opd = path / f'zwfs_SpT={star_SpT}_Vmag={star_Vmag}_dm={dm_shape}_opd.fits'
    fits.writeto(output_opd, opd_map, overwrite=True)

