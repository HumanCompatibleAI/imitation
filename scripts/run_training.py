import yairl.scripts as scripts
import gin.tf
import argparse
import tensorflow as tf


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    scripts.train_and_plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin_config", default='configs/cartpole_orig_airl_repro.gin')
    args = parser.parse_args()

    gin.parse_config_file(args.gin_config)

    main()
