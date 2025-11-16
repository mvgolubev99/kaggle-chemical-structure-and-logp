import importlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Union, Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from utils import _load_cls_from_module

@dataclass
class FpFromFileTransform(BaseEstimator):
    name: str | None = None
    base_dir: str | Path = "./data"
    path: str | Path | None = None
    cash_fp: bool = True

    def _get_fp_path(self):
        if (self.path is None) and self.name and self.base_dir:
            base = Path(self.base_dir)
            fp_paths = base.glob(f"*{self.name}*X_data*.npy")
            if not fp_paths:
                raise FileNotFoundError(f"fingerprint path(s) was/were not found")
            fp_path = sorted(fp_paths)[0]
            return fp_path

        elif self.path and Path(self.path).is_file():
            return self.path

        else:
            raise RuntimeError(
                f"name={self.name}\n"
                f"base_dir={self.base_dir}, exists={Path(self.base_dir).is_dir()}\n"
                f"path={self.path}, exists={Path(self.path).is_file()}"
            )

    def fit(self, X, y=None):
        self.path_ = self._get_fp_path()
        self.X_fp_data = np.load(self.path_) if self.cash_fp else None
        return self

    def transform(self, X):
        if self.X_fp_data is None:
            X_fp_data = np.load(self.path)
        else:
            X_fp_data = self.X_fp_data
        return X_fp_data[X]
    

@dataclass
class PipelineConstructor:
    regressor__model: str | BaseEstimator = "sklearn.dummy.DummyRegressor"
    regressor__model__kwargs: Dict[str, Any] = field(default_factory=dict)

    regressor__fp_transform__name: str | None = None
    regressor__fp_transform__base_dir: str | Path = "./data"
    regressor__fp_transform__path: str | Path | None = None
    regressor__fp_transform__cash_fp: bool = True

    transformer: Optional[BaseEstimator] = field(default_factory=StandardScaler)
    fp_transform_cls: Any = FpFromFileTransform
    pipeline_cls: Any = Pipeline
    transformed_target_regressor_cls: Any = TransformedTargetRegressor

    def get_model_cls(self, model):
        if isinstance(model, str):
            model = _load_cls_from_module(model, return_class_name=False)
            return model
        elif callable(model):
            return model
        else:
            raise TypeError(f"model should be string or callable, but {str(model)} is not")

    def make(self):
        # fp_transform for "regressor"
        fp_transform = self.fp_transform_cls(
            name=self.regressor__fp_transform__name,
            base_dir=self.regressor__fp_transform__base_dir,
            path=self.regressor__fp_transform__path,
            cash_fp=self.regressor__fp_transform__cash_fp,
        )
        # model for "regressor"
        model_cls = self.get_model_cls(self.regressor__model)
        model = model_cls(**self.regressor__model__kwargs)

        # create "regressor"
        regressor_steps = [('fp_transform', fp_transform), ('model', model)]
        regressor = self.pipeline_cls(regressor_steps)

        # create "end-to-end" pipeline using "regressor" and "transformer"
        pipe = self.transformed_target_regressor_cls(
            regressor=regressor,
            transformer=self.transformer
        )
        return pipe

    @classmethod
    def from_kwargs_like_clone_set_params(
            cls, 
            kwargs, 
            prefix_key="regressor__model",
        ):
        extracted_kwargs = dict()
        substracted_kwargs = dict()

        for key, value in kwargs.items():
            if f"{prefix_key}__" in key:
                new_key = key.replace(f"{prefix_key}__", "")
                extracted_kwargs[new_key]=value
            else:
                substracted_kwargs[key]=value
        
        substracted_kwargs[f"{prefix_key}__kwargs"] = extracted_kwargs

        return cls(**substracted_kwargs)

