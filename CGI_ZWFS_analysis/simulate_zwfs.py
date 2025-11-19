from corgisim import scene
from corgisim import instrument
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
from corgisim import outputs
from corgisim.wavefront_estimation import get_drift
import os

def get_optimal_emgain(noiseless_image_hdu):
    """

    Args:
        noiseless_image, numpy.array: c(org)isim psf simulation sim_data

    Returns:
        emgain, float: optimal emgain for emccd tuning

    """
    noiseless_image = noiseless_image_hdu.data
    saturation = 12000
    threshold = 0.8 * saturation
    max = np.nanmax(noiseless_image)
    emgain = threshold/max

    return emgain


def get_clear_pupil(star_properties, frame_exp, bandpass = '1F', dm_case='flat', add_drift=False, outdir=None):
    """

    Args:
        star_properties (dict): A dictionary of the star properties
        frame_exp (float): single frame exposure in seconds
        bandpass (str): The bandpass to use
        dm_case, string: can be set to 'flat', '3e-8' ...
        add_drift, bool: whether to add a drift to the simulation

    Returns:

    """
    if outdir is None:
        script_dir = os.path.dirname(__file__)
        outdir = os.path.join(script_dir, 'data/')
        outdir_noiseless = os.path.join(outdir, 'data/noiseless/')
    else:
        outdir_noiseless = os.path.join(outdir, 'noiseless')

    output_save_file = f'clear_pupil_dm_{dm_case}.fits'
    output_ccd_save_file = f'clear_pupil_dm_{dm_case}_ccd.fits'


    base_scene = scene.Scene(star_properties) #define the astrophysical scene

    if dm_case == 'flat':
        dm_case_name = 'hlc_flat_wfe'
        dm1 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm1_v.fits')
        dm2 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm2_v.fits')

    else:
        dm_case_name = 'hlc_ni_' + dm_case
        dm1 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm1_v.fits')
        dm2 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm2_v.fits')



    optics_keywords = {'cor_type': 'zwfs', 'use_errors': 2, 'polaxis': 10, 'output_dim': 351,
                       'use_fpm': 0, 'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2, 'use_lyot_stop': 0,
                       'use_pupil_lens': 1}

    if add_drift:
        zernike_poly_index, zernike_value_m = get_drift(1, 1, obs='ref', cycle=1)

        optics_keywords.update({'zindex': zernike_poly_index, 'zval_m': zernike_value_m})

    optics = instrument.CorgiOptics('excam', bandpass, optics_keywords=optics_keywords, if_quiet=True,
                                    integrate_pixels=True)

    sim_scene = optics.get_host_star_psf(base_scene)
    outputs.save_hdu_to_fits(sim_scene.host_star_image, outdir=outdir_noiseless, filename=output_save_file,
                             write_as_L1=False)

    ##Tuning the EMCCD
    emgain = get_optimal_emgain(sim_scene.host_star_image)
    if emgain < 1:
        print(f"WARNING: detector saturated with time exposure of {frame_exp}")
        frame_exp *= emgain
        emgain = 1
        print(f"Setting time exposure to {frame_exp} second(s)")

    emccd_keywords = {'em_gain': emgain}
    detector = instrument.CorgiDetector(emccd_keywords)
    sim_scene = detector.generate_detector_image(sim_scene, frame_exp)

    outputs.save_hdu_to_fits(sim_scene.image_on_detector, outdir=outdir, filename=output_ccd_save_file, write_as_L1=False)

    return 1

def get_zwfs_pupil(star_properties, frame_exp, total_exp_time, bandpass = '1F', dm_case='flat', outdir=None):
    if outdir is None:
        script_dir = os.path.dirname(__file__)
        outdir = os.path.join(script_dir, 'data/')
        outdir_noiseless = os.path.join(outdir, 'data/noiseless/')
    else:
        outdir_noiseless = os.path.join(outdir, 'noiseless')

    base_scene = scene.Scene(star_properties) #define the astrophysical scene

    if dm_case == 'flat':
        dm_case_name = 'hlc_flat_wfe'
        dm1 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm1_v.fits')
        dm2 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm2_v.fits')

    else:
        dm_case_name = 'hlc_ni_' + dm_case
        dm1 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm1_v.fits')
        dm2 = proper.prop_fits_read(
            roman_preflight_proper.lib_dir + '/examples/' + dm_case_name + '_dm2_v.fits')

    N_obs = int(total_exp_time / frame_exp)
    N_obs_per_cycle = N_obs // 5

    for cycle in range(1, 6):
        zernike_poly_index, zernike_value_m = get_drift(frame_exp, N_obs_per_cycle, obs='ref', cycle=cycle)
        for n in range(N_obs_per_cycle):
            output_save_file = f'zwfs_pupil_ref_{n}_cycle_{cycle}.fits'
            output_ccd_save_file = f'zwfs_pupil_ref_{n}_cycle_{cycle}_ccd.fits'

            plt.title('WFE')
            plt.xlabel('Zernike noll coeff')
            plt.ylabel('WFE rms (pm)')
            plt.plot(zernike_poly_index, zernike_value_m[n] * 1e12)
            plt.show()

            optics_keywords = {'cor_type': 'zwfs', 'use_errors': 2, 'polaxis': 10, 'output_dim': 351,
                               'use_fpm': 1, 'use_dm1': 1, 'dm1_v': dm1, 'use_dm2': 1, 'dm2_v': dm2,
                               'use_lyot_stop': 0, 'use_pupil_lens': 1, 'zindex': zernike_poly_index,
                               'zval_m': zernike_value_m[n]}

            optics = instrument.CorgiOptics('excam', bandpass, optics_keywords=optics_keywords, if_quiet=True,
                                            integrate_pixels=True)

            sim_scene = optics.get_host_star_psf(base_scene)

            ##Tuning the EMCCD
            emgain = get_optimal_emgain(sim_scene.host_star_image)
            if emgain < 1:
                print(f"WARNING: detector saturated with time exposure of {frame_exp}")
                frame_exp *= emgain
                emgain = 1
                print(f"Setting time exposure to {frame_exp} second(s) and emgain to 1")

            emccd_keywords = {'em_gain': emgain}
            detector = instrument.CorgiDetector(emccd_keywords)
            sim_scene = detector.generate_detector_image(sim_scene, frame_exp)

            ### save products
            outputs.save_hdu_to_fits(sim_scene.host_star_image, outdir=outdir_noiseless, filename=output_save_file,
                                     write_as_L1=False)
            outputs.save_hdu_to_fits(sim_scene.image_on_detector, outdir=outdir, filename=output_ccd_save_file,
                                     write_as_L1=False)
    return 1