import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import pytest

import plottable

class TestPlottable:
    @pytest.fixture(autouse=True)
    def setup(self):
        iris_data = sns.load_dataset("iris")
        self.iris_mean = iris_data.groupby("species").mean()
        self.iris_mean.loc["all"] = iris_data.mean()
        self.iris_std = iris_data.groupby("species").std()
        self.iris_std.loc["all"] = iris_data.std()
        iris_column = pd.MultiIndex.from_arrays(
            (zip(*[label.split("_") for label in self.iris_mean.columns])))
        self.iris_mean.columns = iris_column
        self.iris_std.columns = iris_column

    def test_plot_multiindex(self):
        plottable.plot_table(self.iris_mean)
        assert False
