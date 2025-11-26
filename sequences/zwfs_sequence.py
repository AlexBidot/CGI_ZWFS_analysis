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

from emccd_detect.emccd_detect import EMCCDDetectBase

import pyzelda.zelda as zelda


if __name__ == '__main__':
    __spec__ = None

    root = Path('~/data/Roman/ZWFS/sequences/').expanduser()

    star_Vref = 2.0
    star_Vmag = 2.0

    apply_noise = False
    em_gain     = 1.0
    em_exptime  = 1.0

    wave = 575e-9

    dm_shape = 'flat'
    # dm_shape = 'darkhole'
    # dm_shape = 'lowfs'

    # EMCC noise
    if apply_noise is True:
        full_well_serial = 100000.0         # full well for serial register; 90K is requirement, 100K is CBE
        full_well = 60000.0                 # image full well; 50K is requirement, 60K is CBE
        dark_rate = 0.00056                 # e-/pix/s; 1.0 is requirement, 0.00042/0.00056 is CBE for 0/5 years
        cic_noise = 0.01                    # e-/pix/frame; 0.1 is requirement, 0.01 is CBE
        read_noise = 100.0                  # e-/pix/frame; 125 is requirement, 100 is CBE
        cr_rate = 0                         # hits/cm^2/s (0 for none, 5 for L2)
        bias = 0
        qe = 1.0                            # qe already applied in count rates
        pixel_pitch = 13e-6                 # detector pixel size in meters
        apply_smear = True                  # (LOWFS only) Apply fast readout smear?
        e_per_dn = 1.0                      # post-multiplied electrons per data unit
        nbits = 14                          # ADC bits
        numel_gain_register = 604           # Number of gain register elements
        date = 2028.0                       # decimal year of observation

        emccd = EMCCDDetectBase(em_gain=em_gain, full_well_image=full_well, full_well_serial=full_well_serial,
                                dark_current=dark_rate, cic=cic_noise, read_noise=read_noise, bias=bias,
                                qe=qe, cr_rate=cr_rate, pixel_pitch=pixel_pitch, eperdn=e_per_dn,
                                numel_gain_register=numel_gain_register, nbits=nbits)

        path = root / f'zwfs+{dm_shape}' / 'noiseless'
        npath = path / f'../noisy_Vmag={star_Vmag}_gain={em_gain:.1f}_exptime={em_exptime:.1f}/'
        if not npath.exists():
            npath.mkdir(parents=True, exist_ok=True)

        files = path.glob('*.fits')
        for file in files:
            # read data
            img, hdr = fits.getdata(file, header=True)

            # apply stellar magnitude
            img *= 10**(-(star_Vmag - star_Vref)/2.5)

            # apply EMCCD noise
            img = emccd.sim_sub_frame(img, em_exptime).astype(float)

            # save
            fits.writeto(npath / file.name, img, hdr, overwrite=True)

        path = npath
    else:
        path = root / f'zwfs+{dm_shape}' / 'noiseless'

    # ZELDA analysis
    files = sorted(list(path.glob('psf_ref_*.fits')))
    zelda_pupil_files = [file.stem for file in files]
    clear_pupil_files = ['pupille_dm_flat']
    dark_file = None

    z = zelda.Sensor('ROMAN-CGI')

    clear_pupil, zelda_pupil, center = z.read_files(path, clear_pupil_files, zelda_pupil_files, dark_file,
                                                    collapse_clear=True, collapse_zelda=True, center=(175.5, 175.5))

    opd_map = z.analyze(clear_pupil, zelda_pupil, wave=wave, ratio_limit=5)
    opd_diff = opd_map - np.nanmean(opd_map, axis=0)

    if apply_noise:
        det_suffix = f'_gain={em_gain:.1f}_exptime={em_exptime:.1f}'
    else:
        det_suffix = ''

    output_opd = path / f'../opd_Vmag={star_Vmag}_dm={dm_shape}{det_suffix}.fits'
    fits.writeto(output_opd, opd_map, overwrite=True)

    output_opd_diff = path / f'../opd_diff_Vmag={star_Vmag}_dm={dm_shape}{det_suffix}.fits'
    fits.writeto(output_opd_diff, opd_diff, overwrite=True)

