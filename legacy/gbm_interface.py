# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:05:16 2019

@author: peppe
"""
import json
import pickle
import sqlite3
import warnings

import matplotlib.pyplot as plt
import numpy as np

from legacy._paths import filepath_database, filepath_gbm
from legacy.shared import _LightCurve


class _LightCurveGBM(_LightCurve):
    """
    A class for the GBM lightcurves.
    """

    def __init__(
        self, grb_id, filter_range=(50, 300)
    ):  # TODO: Probably this should not filter on init
        """
        :param grb_id:
        :param filter_range:
        :param trigger_adjust: float, optional. positive value bring the trigger time back of value
        """
        self.id = grb_id
        self.data = self.get_data(grb_id, filter_range)
        for key in self.get_metadata(self.__dict__["id"]).keys():
            setattr(self, key, self.get_metadata(self.__dict__["id"])[key])
        lo_bot, lo_top, hi_bot, hi_top = self.calculate_bkg_interval()
        self.lobckint = lo_bot, lo_top
        self.hibckint = hi_bot, hi_top

    def __repr__(self):
        return "LightCurveGBM['{}']".format(self.id)

    def __getitem__(self, i):
        return np.array(self.data)[i]

    @staticmethod
    def get_data(grb_id, filter_range):
        # open relevant merged grb pickle
        with open(filepath_gbm(grb_id), "rb") as g:
            grb_data = pickle.load(g)

        # here we filter the loaded datafile in some energy band
        loen, hien = filter_range
        grb_data = [
            event[0] for event in grb_data if event[1][0] > loen and event[1][1] < hien
        ]
        return grb_data

    @staticmethod
    def get_metadata(grb_id):
        conn = create_connection()
        answer_dict = {}
        metadata = [
            "id",
            "date",
            "t90",
            "t90_err",
            "tStart",
            "tStop",
            "tTrigger",
            "trig_det",
            "fluence",
            "fluence_err",
            "fluenceb",
            "fluenceb_err",
            "pflx_int",
            "pflx",
            "pflx_err",
            "pflxb",
            "lobckint",
            "hibckint",
        ]
        for label in metadata:
            task = "SELECT {} FROM GBM_GRB WHERE id ='{}'".format(label, grb_id)
            try:
                answer_dict[label] = SQL_query_task(conn, task)[0][0]
            except IndexError:
                raise IndexError("Problems querying {} at id {}.".format(label, grb_id))
        return answer_dict

    def calculate_bkg_interval(self):
        """
        :return: background intervals in mission time
        """

        def retrieve_bck_interval_from_model_metadata():
            lo_bot, lo_top = tuple(
                map(
                    (lambda t: max(min(self.data), t + self.tTrigger)),
                    [float(val) for val in self.lobckint[1:-1].split(", ")],
                )
            )
            hi_bot, hi_top = tuple(
                map(
                    (lambda t: min(max(self.data), t + self.tTrigger)),
                    [float(val) for val in self.hibckint[1:-1].split(", ")],
                )
            )
            if lo_bot == lo_top or hi_bot == hi_top:
                warnings.warn(
                    "\nGRB{} has not enough data for safe background estimation.".format(
                        self.model.id
                    ),
                    stacklevel=2,
                )
            return lo_bot, lo_top, hi_bot, hi_top

        def retrieve_bck_interval_from_model_t90():
            """
            Especially for very long GRBs the bkg intervals from
            metadata refers to times not represented in TTE data lists
            and hence are no good. This routine builds original
            bkg intevals based on the actual available data,
            t90 and trigger time.
            :return: pre-burst bkg lower limit, pre-burst bkg higher limit,
                     post-burst bkg lower limit, post-burst bkg higher limit
            """

            def calculate_margins():
                margin_lo = (self.tTrigger - min(self.data)) / 3
                margin_hi = (max(self.data) - (self.tTrigger + self.t90)) / 5
                return margin_lo, margin_hi

            (
                margin_lo,
                margin_hi,
            ) = (
                calculate_margins()
            )  # we take a margin from trigger for bkg interval estimate
            lo_bot, lo_top = min(self.data) + 1, self.tTrigger - margin_lo
            hi_bot, hi_top = self.tTrigger + self.t90 + margin_hi, max(self.data) - 1
            self.lobckint = lo_bot, lo_top
            self.hibckint = hi_bot, hi_top
            return lo_bot, lo_top, hi_bot, hi_top

        return retrieve_bck_interval_from_model_t90()

    def plot(self, binning=1.0, xlims=None, ylims=None, **kwargs):
        fig = plt.figure(**kwargs)

        data = np.array(self.data) - self.tTrigger
        hist_counts, hist_bin, _ = plt.hist(
            data,
            bins=np.arange(data[0], data[-1] + binning, binning),
            color="#607c8e",
            histtype="step",
        )
        plt.axvline(0, linestyle="dotted", c="orange", linewidth=1, label="tTrigger")
        # plt.axvline(self.tTrigger, linestyle = 'dotted', c = 'lightblue', linewidth = 1, label = 't90')
        plt.title("GRB{}. t90: {:.2f}".format(self.id, self.t90))
        plt.xlabel("Time since trigger [s]")
        if xlims:
            plt.xlim(*xlims)
        if ylims:
            plt.ylim(*ylims)
        plt.ylabel("Counts/{:.3f} s bin".format(binning))
        return fig


def create_connection(path=filepath_database()):
    """
    Create a database connection to the SQLite database
    specified by db_file.

    Args:
        path: path to database file
    Return:
        Connection object or None.
    Raise:
        n/a
    """
    conn = sqlite3.connect(path)
    return conn


def SQL_create_table(conn, create_table_sql):
    """
    Create a table according the create_table_sql statement.

    Args:
        conn: Connection object
        create_table_sql: a CREATE TABLE statement
    Returns:
        n/a
    Raise:
        n/a
    """
    c = conn.cursor()
    c.execute(create_table_sql)


def SQL_insert_data(conn, table, task):
    """
    Starting from a connection item fills a tables according to data

    Args:
        conn: Connection object
        table: Table name
        task: A tuple containing the ordered row data.
    Returns:
        An integer with the ID of the last modified row.
    Raise:
        n/a
    """
    cur = conn.cursor()
    colnum = cur.execute(
        """SELECT count(*) FROM pragma_table_info('GBM_GRB')"""
    ).fetchall()[0][
        0
    ]  # get column number
    sql = (
        """ INSERT INTO """
        + table
        + """ 
              VALUES("""
        + ("?," * colnum)[:-1]
        + """) """
    )
    cur.execute(sql, task)
    print("grb" + task[0] + " added to database.")
    return cur.lastrowid


def SQL_grb_retrieve_trigdet(conn, grb_id):
    """
    This function takes a sqllite connection and a GRB id and
    query the database for the detectors which triggered.
    The detectors ids are given back.

    Args:
        conn: connection object.
        grb_id: id of the GRB.
    Returns:
        triglist_fetch: a list containing the strings id for triggered detectors
    Raise:
        n/a
    """
    cur = conn.cursor()
    triglist_unfetch = json.loads(
        cur.execute("SELECT trig_det FROM GBM_GRB WHERE id =?", (grb_id,)).fetchall()[
            0
        ][0]
    )
    triglist_fetch = [
        int(triglist_unfetch[i][4:6]) for i in range(len(triglist_unfetch))
    ]
    for i, trig in enumerate(triglist_fetch):
        if trig == 10:  # the detectors after 9 are called 'a'..
            triglist_fetch[i] = "a"
        elif trig == 11:  # ..and b.
            triglist_fetch[i] = "b"
    return triglist_fetch


def SQL_query_task(conn, task):
    cur = conn.cursor()
    t90list_fetched = cur.execute(task).fetchall()
    return t90list_fetched


def fetch_grbs(
    num,
    seed=None,
    query="SELECT id, t90, fluence FROM GBM_GRB WHERE t90 > 1 AND t90 < 4. AND pflxb > 2",
):
    import random

    conn = create_connection()
    grb_collection = []
    # QUERY AND PUT ANSWERS CARDS IN A DECK
    deck = SQL_query_task(conn, query)
    # SEED-SHUFFLE THE DECK
    if seed is not None:
        random.seed(seed)
    random.shuffle(deck)
    # PUT AWAY THE FIRST num CARDS
    for card in deck[:num]:
        grb_collection.append(card)
    return grb_collection


def _grb_retrieve_fromdb(grb_id, verbose=False):
    """
    Fetches GRB metadata from GBM sqllite db file.

    Args:
        grb_id: a string containing the GRB id, e.g. something like '080916009'.

    Returns:
        If the fetching ended well the function will return two args:
        1. a tuple containing the GRB metadata and 2. a 0. If the function was
        unable to find a bcat for the requested GRB it will return
        'GRB_directory' and a 1.

    Raises:
        n/a
    """
    conn = create_connection()
    cur = conn.cursor()
    triglist_unfetch = cur.execute(
        "SELECT id,"
        " T90, "
        "T90_err, "
        "tStart, "
        "tStop, "
        "tTrigger, "
        "trig_det, "
        "fluence, "
        "fluence_err, "
        "fluenceb, "
        "fluenceb_err, "
        "pflx_int, "
        "pflx, "
        "pflx_err, "
        "pflxb, "
        "pflxb_err, "
        "lobckint, "
        "hibckint "
        "FROM GBM_GRB WHERE id =?",
        (grb_id,),
    ).fetchall()[0]
    return triglist_unfetch, 0


def query_db_about(grb_id):
    metadata = _grb_retrieve_fromdb(grb_id)[0]
    strings = (
        "id",
        "t90",
        "t90_err",
        "tStart",
        "tStop",
        "tTrig",
        "trigDet",
        "fluence",
        "fluence_err",
        "fluenceb",
        "fluenceb_err",
        "pflx_int",
        "pflx",
        "pflx_err",
        "pflxb",
        "pflxb_err",
        "lobckint",
        "hibckint",
    )
    out = {key: val for (key, val) in list(zip(strings, metadata))}
    return out
