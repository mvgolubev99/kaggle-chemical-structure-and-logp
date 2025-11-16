# FIX SEED!
from utils import set_seed
set_seed(42)

# imports
import argparse
from search_cv import run_experiment_search_cv

def main():
    parser = argparse.ArgumentParser(description="description: run Optuna Search CV using .yaml config")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="./configs/dummy_config.yaml",
        help="path to .yaml config of (optuna) search cv experiment, default is ./configs/dummy_config.yaml", 
    )
    args = parser.parse_args()
    config_path = args.config

    print(f"\nrunning experiment with config: {config_path}\n")

    run_experiment_search_cv(config_path)

if __name__=='__main__':
    main()
