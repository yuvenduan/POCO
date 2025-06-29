from preprocess.generate_sim import save_sim_activity
from preprocess.zebrafish import process_zebrafish_activity, process_zebrafish_ahrens_activity
from preprocess.celegans import celegans_flavell_preprocess, celegans_zimmer_preprocess
from preprocess.mice import mice_preprocess

if __name__ == '__main__':

    # get simulated data
    # save_sim_activity()

    # load and process spontaneous fish activity
    # process_zebrafish_activity()

    for filter_mode in ['none', ]: # No filter used by default, but can add 'lowpass', 'bandpass' version as needed

        process_zebrafish_ahrens_activity(filter_mode=filter_mode)

        celegans_zimmer_preprocess(filter_mode=filter_mode)
        # celegans_flavell_preprocess(filter_mode=filter_mode)

        # mice_preprocess(filter_mode=filter_mode)

        # preprocess_zebrafish_jain(filter_mode=filter_mode)

        continue