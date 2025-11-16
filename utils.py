import random
import importlib
import json
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def make_splits_then_save(
        data,
        output_path="./data/split_info.json", 
        val_size=None, 
        test_size=None,
    ):
    data_ids =  data.index.to_numpy()
    train_ids, test_ids = train_test_split(data_ids, random_state=42, test_size=test_size)
    
    # recalculate val_size
    if isinstance(val_size, float) and isinstance(test_size, float):
        val_size = val_size / (1 - test_size)
    
    train_ids, val_ids = train_test_split(train_ids, random_state=42, test_size=val_size)

    split_info = {
        'train_ids': train_ids.tolist(),
        'val_ids': val_ids.tolist(),
        'test_ids': test_ids.tolist(),
    }

    output_path = Path(output_path)
    if not output_path.parent.is_dir():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
    with open(output_path, "w") as f:
        json.dump(split_info, f)

def split_data_with_saved_indices(
        data, 
        split_info_path="./data/split_info.json",
        target_column='logp',
    ):
    split_info_path=Path(split_info_path)
    if not split_info_path.exists():
        raise FileNotFoundError(f"\'{split_info_path}\' does not exist")
    
    with open(split_info_path, "r") as f:
        split_info = json.load(f)
    
    train_ids = np.array(split_info['train_ids'])
    val_ids = np.array(split_info['val_ids'])
    test_ids = np.array(split_info['test_ids'])

    y_train = data[target_column].iloc[train_ids].values
    y_val = data[target_column].iloc[val_ids].values
    y_test = data[target_column].iloc[test_ids].values

    # that'll 100% guarantee that y_train, y_val and y_test are independent objects
    y_train = copy.deepcopy(y_train)
    y_val = copy.deepcopy(y_val)
    y_test = copy.deepcopy(y_test)

    return (
        (train_ids, y_train), 
        (val_ids, y_val),
        (test_ids, y_test),
    )

def _load_cls_from_module(
        module_dot_class, # for example, 'sklearn.metrics.root_mean_squared_error'
        return_class_name=True
    ):
    module_name, class_name = module_dot_class.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if return_class_name:
        return class_name, cls
    else:
        return cls

def eval_score(
        estimator,
        X,
        y_true,
        metric='sklearn.metrics.root_mean_squared_error',
    ):
    metric_name, scoring = _load_cls_from_module(metric)

    y_pred = estimator.predict(X)
    score = scoring(y_true, y_pred)
    return {
        'metric_name': metric_name,
        'score': float(score),
    }

def bootstrap_score(
        estimator,
        X,
        y_true,
        metric='sklearn.metrics.root_mean_squared_error',
        n_bootstrap=1000,
        confidence_level=0.95,
        random_state=42,
        return_bootstrap_scores=False,
    ):
    metric_name, scoring = _load_cls_from_module(metric)

    n_samples = len(y_true)
    bootstrap_scores = []
    rng = np.random.default_rng(seed=random_state)

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        y_bootstrap = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]

        y_pred = estimator.predict(X_bootstrap)
        score = scoring(y_bootstrap, y_pred)
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)
    mean_score = np.mean(bootstrap_scores)

    # CI from percentiles
    alpha = (1 - confidence_level) / 2
    lower_percentile = 100 * alpha
    upper_percentile = 100 * (1 - alpha)
    conf_interval = np.percentile(bootstrap_scores, [lower_percentile, upper_percentile])

    output_dict = {
        'metric_name': metric_name,
        'score_mean': float(mean_score),
        'score_CI': conf_interval.tolist(),
    }
    
    if return_bootstrap_scores:
        output_dict['bootstrap_scores'] = bootstrap_scores
    
    return output_dict 

def parity_plot(
        estimator,
        X,
        y_true,
        metric='sklearn.metrics.root_mean_squared_error',
        title=None,
        fig_save_path=None,
    ):
    metric_name, scoring = _load_cls_from_module(metric)

    y_pred = estimator.predict(X)
    score = scoring(y_true, y_pred)

    plotter=PredictionErrorDisplay(
        y_true=y_true,
        y_pred=y_pred
    )

    fig, ax = plt.subplots()
    plotter.plot(ax=ax, kind="actual_vs_predicted")
    ax.set_title(
        ',\n'.join([
            f"{title}",
            f"{metric_name}={score:.6f}"
        ])
    )
    if fig_save_path:
        plt.savefig(fname=fig_save_path)
    plt.show()
