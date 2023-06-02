import os


def filepath_gbm(grb_id):
    # return os.path.join('D:/GRB_GBM_pickles/tte_data_' + grb_id + '.pkl')
    # return os.path.join('D:/magazzino/GRB_GBM_pickles/tte_data_' + grb_id + '.pkl')
    return os.path.join(
        os.path.dirname(__file__),
        os.path.join("../gbm_models", "tte_data_") + grb_id + ".pkl",
    )


def filepath_database():
    return os.path.join(os.path.dirname(__file__), "../assets/GBMdatabase.db")


def filepath_instruments(which_file, mode):
    """
    retrieve path of rmf and arf matrices
    :param which_file:
    :param mode:
    :return:
    """
    assert (which_file == "arf" or which_file == "rmf") and (mode == "X" or mode == "S")
    if mode == "S":
        filename = "HERMES-S_onaxis_v3d1." + which_file
    elif mode == "X":
        filename = "HERMES-X_onaxis_v3d1_MLI2layer." + which_file
    # abspath for mixed shitty slash
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../detector_matrices", filename)
    )


def filepath_background_sim(mode):
    """
    retrieve path of background response instrument simulation
    :param mode:
    :return:
    """
    assert mode == "X" or mode == "S"
    if mode == "S":
        filename = "HERMES-S_onaxis_LOWLAT_600_v5d1.bkg"
    elif mode == "X":
        filename = "HERMES-X_onaxis_LOWLAT_600_v5.bkg"
    # abspath for mixed shitty slash
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../detector_matrices", filename)
    )
