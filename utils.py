import random
import importlib
import json
import copy
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

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

def _get_function_args(func):
    """Get all parameter types from function signature"""
    sig = inspect.signature(func)
    
    positional = []
    var_positional = None
    keyword_only = {}
    var_keyword = None
    
    for name, param in sig.parameters.items():
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.POSITIONAL_ONLY):
            positional.append((name, param.default))
        elif param.kind == param.VAR_POSITIONAL:
            var_positional = name
        elif param.kind == param.KEYWORD_ONLY:
            keyword_only[name] = param.default
        elif param.kind == param.VAR_KEYWORD:
            var_keyword = name
    
    return positional, var_positional, keyword_only, var_keyword

def _get_args_kwargs_of_func_from_cfg(func, cfg):
    """
    Extract positional and keyword arguments for a function from a configuration dictionary.
    
    This function analyzes the function's signature and matches parameters with values
    from the configuration dictionary. It handles different parameter types including:
    - Positional and positional-or-keyword parameters
    - Keyword-only parameters  
    - Variable positional parameters (*args)
    - Variable keyword parameters (**kwargs)
    
    Args:
        func (callable): The function whose signature to analyze
        cfg (dict): Configuration dictionary containing parameter values.
                   Keys should match parameter names from the function signature.
                   For *args parameters, use the actual parameter name as key with a list value.
                   For **kwargs parameters, use the actual parameter name as key with a dict value.
    
    Returns:
        tuple: A tuple containing two elements:
            - list: Positional arguments in the correct order, including *args if present
            - dict: Keyword arguments, including **kwargs if present
    
    Raises:
        AttributeError: If a required parameter (without default value) is missing from cfg
        TypeError: If the configuration values don't match the expected parameter types
    
    Example:
        >>> def example_func(a, b=2, *args, c, d=4, **kwargs):
        ...     pass
        >>> cfg = {
        ...     'a': 1,
        ...     'c': 444,
        ...     'args': [10, 20, 30],
        ...     'kwargs': {'x': 100, 'y': 200}
        ... }
        >>> args, kwargs = _get_args_kwargs_of_func_from_cfg(example_func, cfg)
        >>> # args = [1, 2, 10, 20, 30], kwargs = {'c': 444, 'd': 4, 'x': 100, 'y': 200}
    """
    positional, var_positional, keyword_only, var_keyword = _get_function_args(func)

    args = []
    kwargs = {}
    
    # Process positional arguments
    for param_name, default_value in positional:
        if param_name in cfg:
            args.append(cfg[param_name])
        elif default_value is not inspect._empty:
            args.append(default_value)
        else:
            raise AttributeError(f"{param_name} not in cfg dict while no default value exists")
    
    # Process *args from special 'args' key
    if var_positional and 'args' in cfg:
        args.extend(cfg['args'])
    
    # Process keyword-only arguments  
    for param_name, default_value in keyword_only.items():
        if param_name in cfg:
            kwargs[param_name] = cfg[param_name]
        elif default_value is not inspect._empty:
            kwargs[param_name] = default_value
        else:
            raise AttributeError(f"{param_name} not in cfg dict while no default value exists")
    
    # Process **kwargs from special 'kwargs' key
    if var_keyword and 'kwargs' in cfg:
        kwargs.update(cfg['kwargs'])
    
    # Check for extra keys that are not used
    used_keys = (
        {name for name, _ in positional} | 
        set(keyword_only.keys()) | 
        ({'args'} if var_positional else set()) | 
        ({'kwargs'} if var_keyword else set())
    )
    
    extra_keys = set(cfg.keys()) - used_keys
    if extra_keys:
        # Option 1: Ignore silently
        # Option 2: Warn
        print(f"Warning: unused keys in config: {extra_keys}")
        # Option 3: Raise error
        # raise ValueError(f"Unexpected keys in config: {extra_keys}")
    
    return args, kwargs

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

    # Get predictions for entire dataset once
    y_pred = estimator.predict(X)
    
    n_samples = len(y_true)
    bootstrap_scores = []
    rng = np.random.default_rng(seed=random_state)

    for _ in range(n_bootstrap):
        # Sample indices and get corresponding true/pred pairs
        indices = rng.choice(n_samples, n_samples, replace=True)
        
        if hasattr(y_true, 'iloc'):
            y_true_bootstrap = y_true.iloc[indices]
            y_pred_bootstrap = y_pred.iloc[indices] if hasattr(y_pred, 'iloc') else y_pred[indices]
        else:
            y_true_bootstrap = y_true[indices]
            y_pred_bootstrap = y_pred[indices]

        score = scoring(y_true_bootstrap, y_pred_bootstrap)
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
        show_figure=True,
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
    if show_figure:
        plt.show()

    plt.close(fig=fig)
