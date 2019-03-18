from yairl.util.trainer import init_trainer
import gin.tf
import argparse
import tensorflow as tf


@gin.configurable
def main(env_name):
    tf.logging.set_verbosity(tf.logging.INFO)

    trainer = init_trainer(env_name, use_random_expert=False, policy_dir="data")
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin_config", default='configs/cartpole_irl.gin')
    args = parser.parse_args()

    gin.parse_config_file(args.gin_config)

    main()
