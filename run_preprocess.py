from preprocess.generate_sim import save_sim_activity
from preprocess.preprocess import process_zebrafish_activity, process_zebrafish_stim_activity, process_zebrafish_ahrens_activity
from preprocess.celegans import celegans_flavell_preprocess, celegans_zimmer_preprocess
from preprocess.mice import mice_preprocess

if __name__ == '__main__':

    # get simulated data
    # save_sim_activity()

    # load and process spontaneous fish activity
    # process_zebrafish_activity()
    # process_zebrafish_stim_activity()
    process_zebrafish_ahrens_activity()

    # load and process celegans data
    # celegans_zimmer_preprocess()
    # celegans_flavell_preprocess()

    # mice_preprocess()