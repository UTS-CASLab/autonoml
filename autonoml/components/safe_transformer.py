"""
AFETransformer – SafeTransformer wrapped for AutonoML.

* Trains once (base ET ➜ SafeTransformer).
* Afterwards only `.transform()` is used; `.adapt()` is a no‑op.
"""

# ── std / third‑party ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import pyarrow as pa
from SafeTransformer import SafeTransformer
from sklearn.ensemble import ExtraTreesRegressor 

# ── AutonoML core ─────────────────────────────────────────────────────────────
from ..component import MLScaler            # counts as a preprocessing component
from ..data import DataFormatX, DataFormatY


# ──────────────────────────────────────────────────────────────────────────────
# helper utilities
# ──────────────────────────────────────────────────────────────────────────────
def _to_numpy_1d(arr):
    """
    Accept numpy arrays, pandas Series, Arrow ChunkedArray, lists …  
    Always return a **1‑D numpy array** suitable for scikit‑learn.
    """
    # Arrow → numpy
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks().to_numpy()
    # pandas objects
    elif hasattr(arr, "values"):
        arr = arr.values
    # anything else → numpy
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    # ensure 1‑D
    return arr.reshape(-1)


def _to_dataframe(x):
    """Wrap `x` in a pandas DataFrame unless it already is one."""
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)


# ──────────────────────────────────────────────────────────────────────────────
# the actual AutonoML component
# ──────────────────────────────────────────────────────────────────────────────
class AFETransformer(MLScaler):
    # ── construction ──────────────────────────────────────────────────────────
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_model = ExtraTreesRegressor(random_state=0
                                            )

        self.safe_tf   = SafeTransformer(model=self.base_model)

        self.name     += "_AFE"
        self.format_x  = DataFormatX.NUMPY_ARRAY_2D
        self.format_y  = DataFormatY.NUMPY_ARRAY_2D

    # ── required MLComponent API ──────────────────────────────────────────────
    def learn(self, x, y):
        """
        Fit the base SVR first, then SafeTransformer.  
        Both need plain numpy / pandas data, so convert as required.
        """
        # ---------- X ----------
        x_df = _to_dataframe(x)          # DataFrame for SafeTransformer
        x_np = x_df.values               # numpy for SVR

        # ---------- y ----------
        y_np = _to_numpy_1d(y)           # 1‑D numpy for SVR
        y_sr = pd.Series(y_np)           # pandas Series for SafeTransformer

        # ---------- fit ----------
        self.base_model.fit(x_np, y_np)
        self.safe_tf.fit(x_df, y_sr)

    def transform(self, x):
        """Apply the frozen transformer and return a numpy array."""
        x_df       = _to_dataframe(x)
        x_trans_df = self.safe_tf.transform(x_df)
        return x_trans_df.values         # keep downstream format consistent

    # ── incremental learning interface (ignored) ──────────────────────────────
    def adapt(self, x, y):
        """SafeTransformer cannot be updated incrementally – do nothing."""
        return
