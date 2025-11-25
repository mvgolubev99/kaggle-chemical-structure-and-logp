# FIX SEED!
from utils import set_seed
set_seed(42)

# imports
import argparse

from experiments import run_hp_search_experiment

def main():
    parser = argparse.ArgumentParser(description="description: run Optuna Search CV using .yaml config")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="./configs/test_tpe.yaml",
        help="path to .yaml config of optuna search cv experiment, default is ./configs/test_tpe.yaml", 
    )
    args = parser.parse_args()
    config_path = args.config

    exp = run_hp_search_experiment(config_path)

if __name__=='__main__':
    main()
