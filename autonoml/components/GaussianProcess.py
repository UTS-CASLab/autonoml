# autonoml/components/gaussian_process.py
from ..component import MLPredictor   # inheritance helpers
from ..hyperparameter import HPCategorical, HPInt, HPFloat
from ..data import DataFormatX, DataFormatY

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
import numpy as np

class GPRegressor(MLPredictor):
    """
    Scikit-learn Gaussian-Process regressor.
    O(n³) training cost – keep data sizes modest or add a down-sampler!
    """

    # -------- Hyper-parameter space -------------------------------------------------
    @staticmethod
    def new_hpars():
        hpars = dict()

        # kernel choice
        hpars["Kernel"] = HPCategorical(
            in_default="RBF",
            in_options=["RBF", "Matern", "RationalQuadratic"],
            in_info="Type of covariance kernel"
        )
        # length-scale (log-scaled)
        hpars["LengthScale"] = HPFloat(
            in_default=1.0,
            in_min=0.1,
            in_max=10.0,
            is_log_scale=True,
            in_info="Kernel length-scale"
        )
        # observation noise
        hpars["Alpha"] = HPFloat(
            in_default=1e-2,
            in_min=1e-10,
            in_max=1e-1,
            is_log_scale=True,
            in_info="Added diagonal noise (alpha)"
        )
        # optimiser restarts
        hpars["Restarts"] = HPInt(
            in_default=0,
            in_min=0,
            in_max=10,
            in_info="n_restarts_optimizer"
        )
        return hpars

    # -------- Construction ----------------------------------------------------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        kname   = self.hpars["Kernel"].val
        ls      = self.hpars["LengthScale"].val

        if kname == "RBF":
            kernel = RBF(length_scale=ls, length_scale_bounds="fixed")
        elif kname == "Matern":
            kernel = Matern(length_scale=ls, nu=1.5, length_scale_bounds="fixed")
        else:  # RationalQuadratic
            kernel = RationalQuadratic(length_scale=ls, alpha=1.)

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.hpars["Alpha"].val,
            n_restarts_optimizer=self.hpars["Restarts"].val,
            normalize_y=True,
            random_state=0
        )

        self.name += "_GP"
        self.format_x = DataFormatX.NUMPY_ARRAY_2D
        self.format_y = DataFormatY.NUMPY_ARRAY_2D

    # -------- Required API ----------------------------------------------------------
    def learn(self, x, y):               # batch training
        self.model.fit(x, y.ravel())

    def query(self, x):
        """
        Expect x to be a 2-D feature matrix.
        Return predictions as (n_samples, 1) float array.
        """
        y_pred = self.model.predict(x)              # shape (n_samples,)
        return np.asarray(y_pred).reshape(-1, 1)    # <- enforced 2-D

    # -------- Optional: fall-back adaptation (GP isn’t incremental) -----------------
    def adapt(self, x, y):
        # just retrain on everything we’ve got
        self.learn(x, y)
