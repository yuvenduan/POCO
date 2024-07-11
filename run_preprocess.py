from preprocess.generate_sim import save_sim_activity
from preprocess.preprocess import process_spontaneous_activity, process_visual_activity


if __name__ == '__main__':

    # get simulated data
    # save_sim_activity()

    # load and process spontaneous fish activity
    process_spontaneous_activity()

    # load and process fish activity with visual stimulus
    # process_visual_activity()