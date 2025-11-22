# FIX SEED!
from utils import set_seed
set_seed(42)

# imports
import argparse

from experiment_utils import _load_config
from experiments import ExperimentOptunaSearchCV, ExperimentGridSearchCV

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
    cfg = _load_config(config_path)
    
    print(f"\nrunning experiment with config: {config_path}\n")

    if cfg["search"].get("type") == "tpe":
        exp = ExperimentOptunaSearchCV(cfg)
        exp.run()
    
    elif cfg["search"].get("type") == "grid":
        exp = ExperimentGridSearchCV(cfg)
        exp.run()

if __name__=='__main__':
    main()
