import warnings
from pathlib import Path
from typing import Callable
from dataclasses import dataclass

from optuna.samplers import GridSampler, RandomSampler, TPESampler

import experiment_utils

class ExperimentOptunaSearchCV:
    def __init__(self, cfg: str | Path | dict):
        self.cfg = experiment_utils._load_config(cfg)
        experiment_utils._validate_config(self.cfg)

    def _get_sampler(self): 
        return TPESampler(seed=self.cfg["search"].get("seed", 42))

    def _get_objective(self):
        return experiment_utils._get_objective_for_conditional_param_space(self.cfg)
    
    def run(self):
        with warnings.catch_warnings():
            # --- ignore useless warnings ---
            # Optuna throws a warning when dictionary is passed as
            # a choice for trial parameter, but this is the only way to
            # run gridsearch (over all possible cominations) with optuna on a conditional param space.
            # however, these multiple dicts are not really a big problem as they're just a shallow copy
            warnings.filterwarnings(
                "ignore",
                message="Choices for a categorical distribution should be a tuple"
            )
            # lgbm creates column names for features
            # even though no arrays or dataframes are passed as arguments to .fit() method
            # and then throws a warning about X not having feature names
            warnings.filterwarnings(
                "ignore",
                message=r"X does not have valid feature names.*"
            )

            objective = self._get_objective()
            sampler = self._get_sampler()
            self._study = experiment_utils._run_optuna(self.cfg, objective, sampler)
            self.results = experiment_utils._manage_results(self._study, self.cfg)
    

class ExperimentGridSearchCV(ExperimentOptunaSearchCV):
    def __init__(self, cfg):
        super().__init__(cfg)        
        self._all_configs = experiment_utils._flatten_conditional_param_space(self.cfg)
    
    def _get_sampler(self):
        return GridSampler(search_space=self._all_configs)
    
    def _get_objective(self):
        params_sampler = lambda trial, cfg: experiment_utils._get_params_for_trial_from_all_configs(
            trial, cfg, all_configs=self._all_configs)
        return experiment_utils._get_objective_for_conditional_param_space(
            self.cfg, params_sampler=params_sampler)
        





