import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.samplers import GridSampler, RandomSampler, TPESampler
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score, KFold

from pipeline import PipelineConstructor
from utils import (
    _load_cls_from_module,
    bootstrap_score,
    eval_score,
    parity_plot,
    split_data_with_saved_indices,
)


# ---------------
# OptunaSearchCV
# ---------------

@dataclass
class OptunaSearchCV:
    estimator: BaseEstimator
    search_type: str # "grid", "random", "tpe"
    param_space: Optional[Dict[str, Any]] = None
    scoring: str = "neg_root_mean_squared_error"
    n_trials: int = 2
    cv_splits: int = 5
    refit: bool = True
    seed: int = 42
    verbose: int = 1
    experiment_name: str = "unnamed_experiment"
    direction: str = "minimize"

    def _get_sampler(self):
        """Return Optuna sampler based on search_type."""
        if self.search_type == "tpe":
            return TPESampler(seed=self.seed)
        elif self.search_type == "random":
            return RandomSampler(seed=self.seed)
        elif self.search_type == "grid":
            return GridSampler(self.param_space)
        else:
            raise ValueError(f"Unknown search_type: {self.search_type}")

    def _get_objective(self, X, y):
        """
        Generate objective function.
        Objective function calculates cross_val_score with suggested parameters
        on trial.
        """
        cv = KFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.seed
        )

        def objective(trial):
            params = {}
            for name, spec in self.param_space.items():
                # if param spec is list, suggest categorical
                if isinstance(spec, list):
                    params[name] = trial.suggest_categorical(name, spec)

                # if param spec is dict,
                # suggest numerical range (optionally with log scale)
                elif isinstance(spec, dict):
                    t = spec.get("type")
                    low, high = spec["low"], spec["high"]
                    if t == "int":
                        params[name] = trial.suggest_int(name, low, high, log=spec.get("log", False))
                    elif t == "float":
                        params[name] = trial.suggest_float(name, low, high, log=spec.get("log", False))
                    else:
                        raise ValueError(f"Unsupported param type for {name}: {spec}")
                else:
                    raise ValueError(f"Invalid param spec for {name}: {spec}")

            est = clone(self.estimator).set_params(**params)
            scores = cross_val_score(est, X, y, cv=cv, scoring=self.scoring)
            return -np.mean(scores)  # minimize negative metric

        return objective

    def fit(self, X, y):
        """Find best estimator parameters using specified sampler."""
        objective = self._get_objective(X, y)
        sampler = self._get_sampler()

        # create study
        self.study_ = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            study_name=self.experiment_name,
        )

        # optimize
        self.study_.optimize(
            objective,
            n_trials=self.n_trials if self.search_type != "grid" else None,
            show_progress_bar=self.verbose > 0,
        )

        # memorize best params
        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value

        # (optionally)
        # refit estimator with best params on the whole train set
        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)

        # collect results from every trial
        self.results_ = pd.DataFrame(
            [{**t.params, "value": t.value} for t in self.study_.trials]
        )
        return self

    def save_all_results(self, parent_path="./results"):
        path = Path(parent_path) / f"{self.experiment_name}"
        path.mkdir(parents=True, exist_ok=True)
        self.results_.to_csv(
            path / f"{self.experiment_name}_{self.search_type}_search_cv_all_results.csv",
            index=False,
        )

    def save_best_results(self, parent_path="./results"):
        """Save best params and score to YAML file."""
        path = Path(parent_path) / f"{self.experiment_name}"
        path.mkdir(parents=True, exist_ok=True)
        out = {
            "experiment": self.experiment_name,
            "best_score": self.best_score_,
            "best_params": self.best_params_,
        }
        #print(out)
        (path / f"{self.experiment_name}_{self.search_type}_search_cv_best_results.yaml").write_text(yaml.dump(out))

def _get_dataclass_init_args(data_cls):
    return list(data_cls.__annotations__.keys())

class ExperimentSearchCV:
    """
    -------------------
    yaml config example
    -------------------

    experiment_name: dummy_experiment

    data_path: ./data/logP_dataset.csv
    split_info_path: ./data/split_info.json

    results:
      output_path: ./results
      metric: sklearn.metrics.root_mean_squared_error
      save_search_cv_results: True
      save_best_val_results: True
      bootstrap_on_val: True
      display: True
    
    # kwargs for PipelineConstructor
    pipeline:
      regressor__model: sklearn.dummy.DummyRegressor
      regressor__model__kwargs:
        strategy: constant
      regressor__fp_transform__name: MorganFingerprint_2048 # None by default, either name or path should specified
      regressor__fp_transform__base_dir: "./data"   # by default
      regressor__fp_transform__path: None           # by default
      regressor__fp_transform__cash_fp: True        # by default
    
    # kwargs for OptunaSearchCV
    search:
      search_type: str                  # "grid", "random", "tpe"
      param_space:                      # should be specified
        regressor__model__strategy:     # list for categorical parameter 
        - constant
        regressor__model__constant:     # dict for numerical range of parameter
          type: float
          low: 0.25
          high: 2
          log: True
      scoring: "neg_root_mean_squared_error"    # by default
      n_trials: 2                               # by default
      cv_splits: 5                              # by default
      refit: True                               # by default
      seed: 42                                  # by default
      verbose: 1                                # by default
      experiment_name: "unnamed_experiment"     # by default, but forced to use "experiemnt_name" value here
      direction: "minimize"                     # by default

    """
    def __init__(
            self,
            cfg: str | Path | Dict,
            pipeline_constructor_cls: Callable = PipelineConstructor,
            optuna_search_cv_cls: Callable = OptunaSearchCV,
        ):
        self.cfg = self._load_config(cfg)
        self.pipeline_constructor_cls=pipeline_constructor_cls
        self.optuna_search_cv_cls=optuna_search_cv_cls

    def _load_config(self, cfg):
        if isinstance(cfg, str) or isinstance(cfg, Path):
            cfg_path = Path(cfg)
            if not cfg_path.is_file():
                raise FileNotFoundError(f"config file {cfg_path} was not found!")
            cfg_dict = yaml.safe_load(open(cfg_path))
            return cfg_dict
        else:
            # dict is a mutable object!
            return copy.deepcopy(cfg)

    def _make_pipe(self):
        cfg_pipe = self.cfg["pipeline"]
        regressor__model_name = cfg_pipe.get(
            "regressor__model",
            "sklearn.dummy.DummyRegressor"
        )
        model = _load_cls_from_module(
            regressor__model_name,
            return_class_name=False,
        )

        pipe = self.pipeline_constructor_cls(
            regressor__model=model,
            regressor__model__kwargs=cfg_pipe.get("regressor__model__kwargs", {}),
            regressor__fp_transform__name=cfg_pipe.get("regressor__fp_transform__name"),
            regressor__fp_transform__base_dir=cfg_pipe.get("regressor__fp_transform__base_dir", "./data"),
            regressor__fp_transform__path=cfg_pipe.get("regressor__fp_transform__path", None),
            regressor__fp_transform__cash_fp=cfg_pipe.get("regressor__fp_transform__cash_fp", True),
        ).make()
        return pipe

    def _make_search(self, pipe):
        search = self.optuna_search_cv_cls(
            estimator =          pipe,
            search_type =        self.cfg["search"].get("search_type", "grid"),
            param_space =        self.cfg["search"].get("param_space"),
            scoring =            self.cfg["search"].get("scoring","neg_root_mean_squared_error"),
            n_trials =           self.cfg["search"].get("n_trails", 2),
            cv_splits =          self.cfg["search"].get("cv_splits", 5),
            refit =              self.cfg["search"].get("refit", True),
            seed =               self.cfg["search"].get("seed", 42),
            verbose =            self.cfg["search"].get("verbose", 1),
            experiment_name =    self.cfg.get("experiment_name", "unnamed_experiment"),
            direction =          self.cfg["search"].get("direction", "minimize"),
        )
        return search
    
    def _load_n_split_data(self):
        data_path = self.cfg.get("data_path", "./data/logP_dataset.csv")
        data = pd.read_csv(data_path, names = ['smiles', 'logp'])
        return split_data_with_saved_indices(data, target_column="logp")

    def run(self):
        train, val, _ = self._load_n_split_data()
        self.train_ids, self.y_train = train
        self.val_ids, self.y_val = val 

        pipe = self._make_pipe()
        self.search_ = self._make_search(pipe)
        self.search_.fit(X=self.train_ids, y=self.y_train)
        return self

    def _eval_n_save_scoring_results(
            self, 
            scoring_function, 
            savedir, 
            kind="bootstrap_score",
        ):
        results_cfg = self.cfg["results"]

        scoring_results = scoring_function(
            estimator=self.search_.best_estimator_,
            X=self.val_ids,
            y_true=self.y_val,
            metric=results_cfg.get("metric"),
        )
        print(kind, scoring_results)
        if results_cfg.get("save_best_val_results", True):
            scoring_results_path = Path(savedir) / f"{self.cfg["experiment_name"]}_{kind}.json"
            with open(scoring_results_path, "w") as f:
                json.dump(scoring_results, f)

    def generate_results(self):
        results_cfg = self.cfg["results"]

        results_dir = results_cfg.get("output_path")

        if results_dir:
            results_dir = Path(results_dir) / f"{self.cfg["experiment_name"]}"
            results_dir.mkdir(parents=True, exist_ok=True)
        
        # save results using OptunaSearchCV
        if results_cfg.get("save_search_cv_results", True):
            self.search_.save_all_results(parent_path=results_cfg.get("output_path", "./results"))
            self.search_.save_best_results(parent_path=results_cfg.get("output_path", "./results"))
        
        # score on val
        if results_cfg.get("display", True):
            # bootstrap score
            if results_cfg.get("bootstrap_on_val"):
                self._eval_n_save_scoring_results(
                    scoring_function=bootstrap_score,
                    savedir=results_dir,
                    kind="best_on_val_bootstrap_score",
                )

            # eval score
            else:
                self._eval_n_save_scoring_results(
                    scoring_function=eval_score,
                    savedir=results_dir,
                    kind="best_on_val_eval_score",
                )

            # parity plot 
            if results_cfg.get("save_best_val_results", True):
                fig_save_path = results_dir / f"{self.cfg["experiment_name"]}_best_on_val_parity_plot.png"
            else:
                fig_save_path = None

            parity_plot(
                estimator=self.search_.best_estimator_,
                X=self.val_ids,
                y_true=self.y_val,
                metric=results_cfg.get("metric"),
                title=f"{self.cfg["experiment_name"]} best on val",
                fig_save_path=fig_save_path,
            )

def run_experiment_search_cv(config: str | Path | Dict):
    exp = ExperimentSearchCV(config)
    exp.run()
    exp.generate_results()
    return exp

