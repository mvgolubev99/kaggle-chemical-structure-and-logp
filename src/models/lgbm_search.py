from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import yaml

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, PredictionErrorDisplay

from lightgbm import LGBMRegressor

import optuna
from optuna.samplers import TPESampler

from src.chemdata.splits import DatasetSplits


def get_lgbm_params(trial):
    """Get LightGBM parameters for Optuna trial."""
    params = {
        "random_state": 42,
        "verbose": -1,
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
        "subsample": trial.suggest_float("subsample", 0.4, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "gamma": trial.suggest_float("gamma", 0, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10)
    }
    return params


def construct_regressor(model_cls, model_kwargs):
    """Construct a transformed target regressor."""
    return TransformedTargetRegressor(
        regressor=model_cls(**model_kwargs),
        transformer=StandardScaler(),
    )


def bootstrap_score(
        estimator,
        X,
        y_true,
        scoring=mean_absolute_error,
        n_bootstrap=1000,
    ):
    """Calculate bootstrap score for an estimator."""
    y_pred = estimator.predict(X)

    n_samples = len(y_true)
    bootstrap_scores = []
    rng = np.random.default_rng(seed=42)

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, n_samples, replace=True)
        score = scoring(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)

    return np.mean(bootstrap_scores)


class FingerprintModelSearch:
    """Optuna search for fingerprint-based models."""
    
    def __init__(self, splits: DatasetSplits, fp_name: str, model_cls, get_params):
        """
        Initialize the search.
        
        Args:
            splits: DatasetSplits object containing data
            fp_name: Name of fingerprint to use
            model_cls: Model class (e.g., LGBMRegressor)
            get_params: Function to generate parameters for trial
        """
        self.splits = splits
        self.fp_name = fp_name
        self.model_cls = model_cls
        self.get_params = get_params

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and validation data."""
        X_train = self.splits.X_data[self.fp_name][self.splits.train_ids]
        X_val = self.splits.X_data[self.fp_name][self.splits.val_ids]
        return X_train, self.splits.y_train, X_val, self.splits.y_val

    def run(self, n_trials: int = 20) -> Dict[str, Any]:
        """
        Run Optuna optimization.
        
        Args:
            n_trials: Number of trials to run
            
        Returns:
            Dictionary with optimization results
        """
        X_train, y_train, X_val, y_val = self._prepare_data()

        def objective(trial):
            params = self.get_params(trial)
            trial.set_user_attr("params", params)

            reg = construct_regressor(self.model_cls, params)
            reg.fit(X_train, y_train)

            score = bootstrap_score(
                estimator=reg,
                X=X_val,
                y_true=y_val,
            )
            return score

        study_name = f"{self.model_cls.__name__}_{self.fp_name}"
        study = optuna.create_study(
            sampler=TPESampler(seed=42),
            study_name=study_name,
            direction="minimize",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_trial.user_attrs["params"]
        best_score = float(study.best_value)

        print("\n" + "-" * 64)
        print(f"study '{study_name}' finished after {n_trials} trials\n")
        print(yaml.dump({"best_params": best_params}))
        print(f"best_score: {best_score}\n")

        return {
            "study_name": study_name,
            "fp_name": self.fp_name,
            "model_cls": self.model_cls.__name__,
            "best_params": best_params,
            "best_score": best_score,
        }


def prepare_train_data_from_study_results(results: Dict[str, Any], splits: DatasetSplits, 
                                         fit_on_subset: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """Prepare training data from study results."""
    fp_name = results["fp_name"]

    if fit_on_subset == "train":
        X = splits.X_data[fp_name][splits.train_ids]
        y = splits.y_train

    elif fit_on_subset == "train_val":
        train_val_ids = np.concatenate((splits.train_ids, splits.val_ids))
        X = splits.X_data[fp_name][train_val_ids]
        y = np.concatenate((splits.y_train, splits.y_val))

    else:
        raise ValueError(f"fit_on_subset has incorrect value: {fit_on_subset}")

    return X, y


def prepare_test_data_from_study_results(results: Dict[str, Any], splits: DatasetSplits, 
                                        score_on: str = "val") -> Tuple[np.ndarray, np.ndarray]:
    """Prepare test data from study results."""
    fp_name = results["fp_name"]

    if score_on == "val":
        X = splits.X_data[fp_name][splits.val_ids]
        y = splits.y_val

    elif score_on == "test":
        X = splits.X_data[fp_name][splits.test_ids]
        y = splits.y_test

    else:
        raise ValueError(f"score_on has incorrect value: {score_on}")

    return X, y


def construct_regressor_from_study_results(results: Dict[str, Any]) -> TransformedTargetRegressor:
    """Construct regressor from study results."""
    params = results["best_params"]
    fp_name = results["fp_name"]

    model_cls_name = results["model_cls"]
    if model_cls_name == "LGBMRegressor":
        model_cls = LGBMRegressor
    else:
        raise ValueError(f"unknown model_cls {model_cls_name}")

    return construct_regressor(model_cls, model_kwargs=params)


def rerun_fit_score_from_study_results(
        results: Dict[str, Any],
        splits: DatasetSplits,
        fit_on_subset: str = "train",
        score_on: str = "val",
    ) -> Tuple[TransformedTargetRegressor, float]:
    """
    Refit and score model from study results.
    
    Args:
        results: Study results dictionary
        splits: DatasetSplits object
        fit_on_subset: Subset to fit on ('train' or 'train_val')
        score_on: Subset to score on ('val' or 'test')
        
    Returns:
        Tuple of (fitted_regressor, score)
    """
    X_train, y_train = prepare_train_data_from_study_results(results, splits, fit_on_subset)
    X_test, y_test = prepare_test_data_from_study_results(results, splits, score_on)

    regressor = construct_regressor_from_study_results(results)
    regressor.fit(X_train, y_train)

    score = bootstrap_score(
        estimator=regressor,
        X=X_test,
        y_true=y_test,
    )

    y_test_pred = regressor.predict(X_test)

    plotter = PredictionErrorDisplay(
        y_true=y_test,
        y_pred=y_test_pred
    )

    fig, ax = plt.subplots()
    plotter.plot(ax=ax, kind="actual_vs_predicted")
    ax.set_title(
        ',\n'.join([
            f"best regressor from study: '{results['study_name']}'",
            f"fit_on_subset: '{fit_on_subset}'",
            f"score_on: '{score_on}'",
            f"score(MAE)={score:.6f}",
        ])
    )

    plt.show()

    return regressor, score