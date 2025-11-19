from CGI_ZWFS_analysis.simulate_zwfs import *

if __name__ == '__main__':
    vmag_ref = 2
    vmag_sci = 0
    sptype = 'G0V'

    x_off_mas = 300
    y_off_mas = 0

    x_off_star_mas = 0 #not supported yet, set to 0
    y_off_star_mas = 0 #not supported yet, set to 0
    dmag = 25

    frame_exp = 1  # sec
    total_exp_time = 20  # sec

    ref_star_properties = {'Vmag': vmag_ref, 'spectral_type': sptype, 'magtype': 'vegamag',
                                'position_x': x_off_star_mas, 'position_y': y_off_star_mas}


    #get_clear_pupil(ref_star_properties, frame_exp)
    get_zwfs_pupil(ref_star_properties, frame_exp, total_exp_time)

    #TODO add Z1 and Z2 jitter + also generate offseted PSF ~ 5 mas