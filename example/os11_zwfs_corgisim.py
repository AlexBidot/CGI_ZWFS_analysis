import proper
import roman_preflight_proper
import sys
from pathlib import Path

try:
    from CGI_ZWFS_analysis.simulate_zwfs import *
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).parent.parent))
    from CGI_ZWFS_analysis.simulate_zwfs import *

if __name__ == '__main__':
    # Copy the default prescription file
    roman_preflight_proper.copy_here()

    vmag_ref = 2
    vmag_sci = 0
    sptype = 'G0V'

    x_off_mas = 300
    y_off_mas = 0

    x_off_star_mas = 200 #not supported yet, set to 0
    y_off_star_mas = 0 #not supported yet, set to 0
    dmag = 25

    frame_exp = 1  # sec
    total_exp_time = 20  # sec

    ref_star_properties = {
        'Vmag': vmag_ref, 'spectral_type': sptype, 'magtype': 'vegamag',
        'position_x': x_off_star_mas, 'position_y': y_off_star_mas
        }

    outdir = Path(__file__).parent.parent / 'data'

    optics_keywords = {'source_x_offset_mas': 30, 'source_y_offset_mas': 0}

    #get_clear_pupil(ref_star_properties, frame_exp)
    get_zwfs_pupil(ref_star_properties, frame_exp, total_exp_time, optics_keywords=optics_keywords, outdir=outdir, plot=False)

    #TODO add Z1 and Z2 jitter + also generate offseted PSF ~ 5 mas