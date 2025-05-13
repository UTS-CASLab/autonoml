"""
Kernel-PCA pre-processor for AutonoML
Author: Dilshan Sonnadara
"""

# --- AutonoML base classes / helpers ----------------------------------------
from ..hyperparameter import HPCategorical, HPInt, HPFloat
from ..component      import MLScaler         # << we only transform X
from ..data           import DataFormatX, DataFormatY
# ---------------------------------------------------------------------------

from sklearn.decomposition import KernelPCA         # you already have sklearn

class KPCA(MLScaler):
    """
    Kernel‐PCA wrapper.  Produces a low-dim representation of X before it
    reaches the downstream predictor.
    """

    # ---------- hyper-parameter definitions ---------------------------------
    @staticmethod
    def new_hpars():
        """
        Return a dict {<name>: Hyperparameter-object} describing the search
        space.  These names will be the keys of self.hpars.
        """
        h = dict()

        # categorical kernel choice
        h["kernel"] = HPCategorical(
            in_options = ["linear", "poly", "rbf", "sigmoid", "cosine"],
            in_default = "rbf",
            in_info    = "KPCA kernel function"
        )

        # number of components   –  sample on a *logarithmic* int grid
        h["n_components"] = HPInt(
            in_default = 10,
            in_min     = 2,
            in_max     = 200,
            is_log_scale = True
        )

        # gamma only matters for rbf/poly/sigmoid
        h["gamma"] = HPFloat(
            in_default   = 0.1,
            in_min       = 1e-4,
            in_max       = 10.0,
            is_log_scale = True
        )

        return h
    # ------------------------------------------------------------------------

    # ------------ constructor -----------------------------------------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Build the wrapped sklearn object from *current* values of self.hpars
        self._build_model()

        # ► book-keeping                                                         ┐
        self.name     += "_KPCA"                                                 #│
        self.format_x  = DataFormatX.NUMPY_ARRAY_2D
        self.format_y  = DataFormatY.NUMPY_ARRAY_2D

    # ------------------------------------------------------------------------
    # internal helper to (re)create KPCA after a hyper-param change
    def _build_model(self):
        hp = self.hpars
        self.model = KernelPCA(
            kernel       = hp["kernel"].val,
            n_components = hp["n_components"].val,
            gamma        = hp["gamma"].val,
            fit_inverse_transform = False,
            copy_X       = True,
            n_jobs       = 1,                       # AutonoML handles parallelism
            random_state=0
        )

    # ------------------------------------------------------------------------
    # When BOHB samples new hyper-params AutonoML will call `update_hpars`
    # (inherited).  We just need to rebuild the sklearn object afterwards.
    def post_update_hpars(self):
        self._build_model()

    # ------------------------------------------------------------------------
    # Mandatory MLPreprocessor API
    def learn(self, x, y):
        """Fit the KPCA on training data (y ignored)."""
        self.model.fit(x)

    def transform(self, x):
        """Apply the learnt projection."""
        return self.model.transform(x)

    # (query() is not needed for a pure pre-processor)
