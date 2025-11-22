# config for tpe search

```yaml
experiment_name: my_experiment_name

search:
  type: tpe
  scoring: neg_root_mean_squared_error
  n_trials: 96                              
  cv_splits: 5
  refit: true
  seed: 42
  verbose: 0  
  direction: minimize

results:
  output_path: ./results
  metric: sklearn.metrics.root_mean_squared_error
  bootstrap_on_val: true
  display_parity_plot: true
  
param_space:
  data_representation:
    fp_1: {fp_name_or_path: MorganFingerprint_2048}
    fp_2: {fp_name_or_path: FeaturesMorganFingerprint_2048}
  model:
    module_name.model_class_1:
      alpha: {type: float, low: 0.001, high: 1.0, log: true}
      flag: [true, false]
    module_name.model_class_2:
      lambda: {type: float, low: 0.0, high: 1.0, log: false}
      category: [A, B, C]
```

# for every trial build simple cfg:

```yaml
data_representation:
  fp_name_or_path: MorganFingerprint_2048
  # here may be other keyword arguments
model: module_name.model_class_1
model_kwargs:
  alpha: 0.1
  flag: true
```