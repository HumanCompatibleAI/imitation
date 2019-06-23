import datetime
import glob
import os.path

import tensorflow as tf


def make_summary_writer(exp_name="AIRL", graph=None):
    summary_base = os.path.join("output/", exp_name, "summary/")
    today_str = datetime.datetime.today().strftime('%Y-%m-%d')
    dir_list = glob.glob(os.path.join(summary_base, today_str + "*/"))

    i = 0
    done = False
    run_name = None
    while not done:
        run_name = today_str + "_run{}/".format(i)
        run_dir = os.path.join(summary_base, run_name)
        done = run_dir not in dir_list
        i += 1

    tf.logging.info("building summary directory at " + run_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    summary_writer = tf.summary.FileWriter(run_dir, graph=graph)
    return summary_writer
