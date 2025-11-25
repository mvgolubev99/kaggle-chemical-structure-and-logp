import copy
import yaml
import itertools
from pathlib import Path
from typing import Tuple, Dict, Any, List, Callable

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.compose import TransformedTargetRegressor

import utils
from preprocessing import _load_and_preprocess

_load_cls = lambda name: utils._load_cls_from_module(
    module_dot_class=name, 
    return_class_name=False,
)

def _load_config(cfg):
    if isinstance(cfg, str) or isinstance(cfg, Path):
        cfg_path = Path(cfg)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"config file {cfg_path} was not found!")
        cfg_dict = yaml.safe_load(open(cfg_path))
        return cfg_dict
    else:
        # dict is a mutable object!
        return copy.deepcopy(cfg)

def _validate_config(cfg):
    # validate first layer
    cfg_keys = [
        "experiment_name",
        "search",
        "results",
        "param_space",
    ]
    for key in cfg_keys:
        if key not in cfg.keys():
            raise KeyError(f"config validation failed: no \'{key}\' in cfg")
    
    # validate experiment name
    experiment_name = cfg["experiment_name"]
    if (experiment_name is None) or not isinstance(experiment_name, str):
        raise ValueError(f"invalid experiment name: {experiment_name}")
    
    # validate search type
    search_type = cfg["search"].get("type")
    if search_type is None:
        raise KeyError(
            f"type of search was not specified. choose, for example, \'tpe\' or \'grid\'")
    
    supported_search_types = ['tpe', 'grid']
    if search_type not in supported_search_types:
        raise ValueError(f"invalid search type, supported values: {supported_search_types}")
    
    # validate param space
    param_space = cfg["param_space"]
    if not isinstance(param_space, dict):
        raise TypeError(f"param_space is of \'{type(param_space)}\' type but dict is expected")
    
    for key in ["data_representation", "model"]:
        if key not in param_space.keys():
            raise KeyError(f"param_space does not have \'{key}\' key")

    # validate param space, data_representation
    data_representation = param_space["data_representation"]
    
    if not isinstance(data_representation, dict):
        raise TypeError(f"data_representation cfg is of \'{type(data_representation)}\' type but dict is expected")

    if len(data_representation) == 0:
        raise ValueError(f"data_representation cfg should have at least one key and value")
    
    for key, value in data_representation.items():
        if not isinstance(value, dict):
            raise TypeError(f"data_representation key \'{key}\' has value of \'{type(value)}\' but dict is expected")
        
        if len(value) == 0:
            raise ValueError(
                f"data_representation key \'{key}\' should have at least one key and value\n"
                f"at least fp_name_or_path should be specified"
            )
    
    # validate param space, model
    model = param_space["model"]

    if not isinstance(model, dict):
        raise TypeError(f"model cfg is of \'{type(model)}\' but dict is expected")
    
    for key, value in model.items():
        if not isinstance(value, dict):
            raise TypeError(
                f"cfg[\"param_space\"][\"model\"] key \'{key}\' have value with \'{type(value)}\' "
                f"but dict is expected. Even if model is called without any kwargs, you should specify it "
                f"as empty dictionary"
            )
        
        # validate parameter specifications    
        for name, spec in value.items():
            # check if all parameter specs ar passed as list for grid search
            if search_type == "grid" and not isinstance(spec, list):
                raise TypeError(
                    f"search type is set to grid, but "
                    f"values for parameter \'{name}\' of \'{key}\' are not passed as list!"
                )

            # only numerical ranges can be specified as dict
            if isinstance(spec, dict):
                spec_keys = ["type", "low", "high"]
                for spec_key in spec_keys:
                    if spec_key not in spec.keys():
                        raise ValueError(
                            f"parameter \'{name}\' of \'{key}\' model does not have \'{spec_key}\' key "
                            f"in specification dict"
                        )
                
                log_flag = spec.get("log")
                if (log_flag is not None) and not isinstance(log_flag, bool):
                    raise TypeError(
                        f"log flag for parameter \'{name}\' of \'{key}\' model should be bool, "
                        f"not {type(log_flag)}"
                    )
                

def _load_and_preprocess_data_from_cfg(cfg):
    args, kwargs = utils._get_args_kwargs_of_func_from_cfg(
        func = _load_and_preprocess,
        cfg=cfg,
    )
    data = _load_and_preprocess(*args, **kwargs)
    return data

def _load_and_configure_regressor(
        model, 
        model_kwargs, 
        y_scaler='sklearn.preprocessing.StandardScaler',
    ):
    model_cls = _load_cls(model)
    model_instance = model_cls(**model_kwargs)

    y_scaler_cls = _load_cls(y_scaler)
    y_scaler_instance = y_scaler_cls()

    transformed_target_regressor = TransformedTargetRegressor(
        regressor=model_instance,
        transformer=y_scaler_instance,
    )

    return transformed_target_regressor
        

def _get_params_for_trial_from_cfg(trial, cfg):
    """
    Samples data representation, model and model parameters
    from given (conditional) space of parameters in experiment config 
    """
    # ---------------------------------------------------------
    # Suggest data representation
    # ---------------------------------------------------------

    # get subconfig for data representations
    data_configs = cfg["param_space"].get(
        "data_representation", 
        {"MorganFingerprint_2048": {"fp_name_or_path": "MorganFingerprint_2048"}}
    )
    # sample data representation from list of available data representations
    suggested_data_representation = trial.suggest_categorical(
        name="data_representation", 
        choices=list(data_configs.keys()),
    )

    # store chosen data representation name as value for "data_representation" key in params
    suggested_params = {"data_representation": data_configs[suggested_data_representation]}
    
    # ---------------------------------------------------------
    # Suggest model
    # ---------------------------------------------------------

    # get subconfig for models
    model_configs = cfg["param_space"].get(
        "model",
        {"sklearn.dummy.DummyRegressor": {"strategy": "median"}},
    )
    # sample model from list of available models
    suggested_model = trial.suggest_categorical(
        name="model",
        choices=list(model_configs.keys()),
    )
    # store chosen model name as value for "model" key in params
    suggested_params["model"] = suggested_model
    suggested_params["model_kwargs"] = {}

    # ---------------------------------------------------------
    # Suggest parameters for model
    # ---------------------------------------------------------

    # get subconfig for paramaters of chosen models
    suggested_model_param_space = model_configs[suggested_model]

    for name, spec in suggested_model_param_space.items():
        # if param spec is list, suggest categorical
        if isinstance(spec, list):
            suggested_params["model_kwargs"][name] = trial.suggest_categorical(
                f"{str(suggested_model)}__{name}", 
                spec,
            )

        # if param spec is dict,
        # suggest numerical range (optionally with log scale)
        elif isinstance(spec, dict):
            t = spec.get("type")
            low, high = spec["low"], spec["high"]
            if t == "int":
                suggested_params["model_kwargs"][name] = trial.suggest_int(
                    f"{str(suggested_model)}__{name}", 
                    low, 
                    high, 
                    log=spec.get("log", False),
                )
            elif t == "float":
                suggested_params["model_kwargs"][name] = trial.suggest_float(
                    f"{str(suggested_model)}__{name}", 
                    low, 
                    high, 
                    log=spec.get("log", False),
                )
            else:
                raise ValueError(f"Unsupported param type for {name}: {spec}")
        else:
            suggested_params["model_kwargs"][name] = spec
        
    return suggested_params

def _flatten_conditional_param_space(cfg):
    """
    Expand a conditional parameter space into a fully enumerated
    list of all valid configurations.

    Input format (example):
        param_space:
        data_representation:
            fp_1: {fp_name_or_path: MorganFingerprint_2048}
            fp_2: {fp_name_or_path: FeaturesMorganFingerprint_2048}
        model:
            module_name.ModelA:
            alpha: [0, 1]
            flag: [true, false]
            module_name.ModelB:
            lambda: [0, 1]
            category: [A, B, C]

    Output format:
        {
            "params": [
                {
                    "data_representation": {
                        "fp_name_or_path": "MorganFingerprint_2048"
                        # possibly other fp kwargs
                    },
                    "model": "module_name.ModelA",
                    "model_kwargs": {
                        "alpha": 0,
                        "flag": true
                    }
                },
                ...
            ]
        }

    The returned configurations are "flat" and ready to be passed into
    `_load_and_configure_regressor` directly.
    """
    param_space = cfg["param_space"]
    fp_space = param_space.get("data_representation", {})
    model_space = param_space.get("model", {})

    all_configs = []

    # Iterate over fingerprint choices
    for fp_name, fp_kwargs in fp_space.items():

        # fp_kwargs is already a dict of keyword args for fp
        # There is NO grid product on fp_kwargs, because
        # these arguments are fixed per fp key.
        #
        # But if in the future some fp args need grid product,
        # it can be extended here.
        #
        # For now: it's simply a single fixed dict.

        fp_config_dict = fp_kwargs  # already a dict: {fp_name_or_path: ...}

        # Iterate over models
        for model_full_name, model_params in model_space.items():

            # model_params is a dict like:
            #   {"alpha": [0,1], "flag": [true,false]}
            # We need Cartesian product over its values.
            model_param_names = list(model_params.keys())
            model_param_values = list(model_params.values())

            # Cartesian product of parameter value lists
            for combo in itertools.product(*model_param_values):
                model_kwargs = dict(zip(model_param_names, combo))

                # Build final configuration
                config = {
                    "data_representation": fp_config_dict.copy(),
                    "model": model_full_name,
                    "model_kwargs": model_kwargs,
                }

                all_configs.append(config)

    return {"params": all_configs}

def _get_params_for_trial_from_all_configs(trial, cfg, all_configs):
    params = trial.suggest_categorical(
        name="params", 
        choices=all_configs["params"],
    )
    return params

def _get_objective_for_conditional_param_space(
        cfg, 
        params_sampler = _get_params_for_trial_from_cfg, 
    ):    
    def objective(trial):
        params = params_sampler(trial, cfg)
        trial.set_user_attr("params", params)

        data = _load_and_preprocess_data_from_cfg(
            cfg=params["data_representation"]
        )
        X_train, y_train = data["train"]

        model = _load_and_configure_regressor(
            model=params["model"],
            model_kwargs=params["model_kwargs"],
        )

        cv_splits = cfg["search"].get("cv_splits", 5)
        if cv_splits >= 2:
            cv = KFold(
                n_splits=cfg["search"].get("cv_splits", 5),
                shuffle=True,
                random_state=cfg["search"].get("seed", 42)
            )

            scores = cross_val_score(
                estimator=model, 
                X=X_train, 
                y=y_train, 
                cv=cv, 
                scoring=cfg["search"].get("scoring", "neg_root_mean_squared_error"),
            )
            return -np.mean(scores)  # minimize negative metric
        
        else:
            X_train_, X_val_, y_train_, y_val_ = train_test_split(
                X_train, y_train,
                random_state=42,
                shuffle=True,
            )
            model.fit(X=X_train_, y=y_train_)
            
            if cfg["search"].get("bootstrap_without_cv"):
                bootstrap_mean_score = utils.bootstrap_score(
                    estimator=model,
                    X=X_val_,
                    y_true=y_val_,
                    metric=cfg["results"].get("metric", "sklearn.metrics.root_mean_squared_error")
                ).get("score_mean")
                return bootstrap_mean_score
            else:
                score = utils.eval_score(
                    estimator=model,
                    X=X_val_,
                    y_true=y_val_,
                    metric=cfg["results"].get("metric", "sklearn.metrics.root_mean_squared_error")
                ).get("score")
                return score
                
    return objective

def _run_optuna(
        cfg: Dict, 
        objective: Callable,
        sampler: optuna.samplers.BaseSampler,
    ):
    # create study
    study = optuna.create_study(
        direction=cfg["search"].get("direction", "minimize"),
        sampler=sampler,
        study_name=cfg["experiment_name"],
    )

    # set n_trials for optimization
    if cfg["search"].get("type") != "grid":
        n_trials = cfg["search"].get("n_trials", 2)
    else: 
        n_trials = None

    # optimize
    study.optimize(
        objective,
        n_trials = n_trials,
        show_progress_bar=cfg["search"].get("verbose", 0) > 0,
    )

    return study

def _manage_results(study, cfg):
    # extract params/cv_score from best trial
    best_params = study.best_trial.user_attrs["params"]
    best_cv_score = study.best_value

    best_params["from_experiment"] = cfg["experiment_name"]
    best_params["cv_score"] = {
        "scoring": cfg["search"].get("scoring"),
        "value": best_cv_score,
    }

    # extract info from every trial
    all_trials_results = pd.DataFrame(
            [{**t.params, "value": t.value} for t in study.trials]
        )

    # (optional) refit regressor with best params on the whole train set
    refit = cfg["results"].get("refit", True) 
    if refit:
        # load data from files or global cache
        data  = _load_and_preprocess_data_from_cfg(cfg=best_params["data_representation"])
        X_train, y_train = data["train"]
        X_val, y_val = data["val"]

        # configure and fit regressor
        best_regressor = _load_and_configure_regressor(
            model=best_params["model"],
            model_kwargs=best_params["model_kwargs"],
        )
        best_regressor.fit(X=X_train, y=y_train)

        # evaluate score on validation dataset either with or without metric bootstraping
        if cfg["results"].get("bootstrap_on_val", True):
            scoring_function = utils.bootstrap_score
        else:
            scoring_function = utils.eval_score

        val_score_results = scoring_function(
            estimator=best_regressor,
            X=X_val,
            y_true=y_val,
            metric=cfg["results"].get("metric", "sklearn.metrics.root_mean_squared_error"),
        )

        # store validation score results in best_params dict
        best_params["val_score"] = val_score_results

    # if output_path is not None,
    # create in this directory subdirectory with experiment name
    # then save best_params dict in .yaml format 
    # and all_trials_results dataframe in .csv format 
    output_path = cfg["results"].get("output_path") 
    if output_path:
        output_path = Path(output_path) / f"{cfg["experiment_name"]}"
        output_path.mkdir(parents=True, exist_ok=True)
        best_params_output_path = output_path / f"{cfg["experiment_name"]}_best_params.yaml"
        best_params_output_path.write_text(yaml.dump(best_params))
        
        all_trials_results_output_path = output_path / f"{cfg["experiment_name"]}_all_trials_results.csv"
        all_trials_results.to_csv(
            all_trials_results_output_path,
            index=False,
        )
    
    # If output_path is specified and/orr display_parity_plot is set to True
    # save and/or display parity plot
    if (output_path is not None) or cfg["results"].get("display_parity_plot", True):
        if output_path is None:
            fig_save_path = None
        else:
            fig_save_path = output_path / f"{cfg["experiment_name"]}_best_on_val_parity_plot.png"

        utils.parity_plot(
            estimator=best_regressor,
            X=X_val,
            y_true=y_val,
            metric=cfg["results"].get("metric", "sklearn.metrics.root_mean_squared_error"),
            title=f"{cfg["experiment_name"]} best regressor on validation",
            fig_save_path=fig_save_path,
            show_figure=cfg["results"].get("display_parity_plot", True),
        )

    # print best params with best_cv_score and val_score 
    print("\n" + "-"*20)
    print("best params with best_cv_score and val_score:\n")
    print(yaml.dump(best_params))

    # collect output
    results = {
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'val_score': val_score_results,
        'all_trials_results': all_trials_results, 
    } 

    # if refit is set to True, also return best_regressor fitted on the whole train set
    if refit:
        results["best_regressor"] = best_regressor
    
    return results
