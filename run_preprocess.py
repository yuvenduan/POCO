from preprocess.generate_sim import save_sim_activity
from preprocess.preprocess import process_spontaneous_activity, process_visual_activity, process_stim_activity
from preprocess.celegans import celegans_flavell_preprocess, celegans_zimmer_preprocess
from preprocess.mice import mice_preprocess

if __name__ == '__main__':

    # get simulated data
    # save_sim_activity()

    # load and process spontaneous fish activity
    # process_spontaneous_activity()

    # load and process fish activity with optogenetic stimulation
    # process_stim_activity()

    # load and process celegans data
    # celegans_zimmer_preprocess()
    celegans_flavell_preprocess()

    # load and process fish activity with visual stimulus
    # process_visual_activity()

    # mice_preprocess()