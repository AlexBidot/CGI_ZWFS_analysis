import proper
import roman_preflight_proper
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

from pathlib import Path

try:
    import CGI_ZWFS_analysis.simulate_zwfs as zwfs
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).parent.parent))
    import CGI_ZWFS_analysis.simulate_zwfs as zwfs

if __name__ == '__main__':
    __spec__ = None

    # Copy the default prescription file
    roman_preflight_proper.copy_here()

    vmag_ref = 6
    vmag_sci = 0
    sptype = 'G0V'

    x_off_mas = 300
    y_off_mas = 0

    frame_exp = 1  # sec
    total_exp_time = 20  # sec

    ref_star_properties = {
        'Vmag': vmag_ref, 'spectral_type': sptype, 'magtype': 'vegamag',
        }

    outdir = Path(__file__).parent.parent / 'data'

    # zwfs.get_clear_pupil(ref_star_properties, frame_exp)
    zwfs.get_zwfs_pupil(ref_star_properties, frame_exp, total_exp_time, outdir=outdir, plot=True)

    #TODO add Z1 and Z2 jitter + also generate offseted PSF ~ 5 mas