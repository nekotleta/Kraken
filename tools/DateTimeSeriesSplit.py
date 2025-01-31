import numpy as np
import pandas as pd
from typing import Tuple, Optional, Iterator, Any


class DateTimeSeriesSplit:
    """Class for creating a time series split for a pandas dataframe with a datetime column"""

    def __init__(
            self,
            n_splits: int = 4,
            test_size: int = 1,
            margin: int = 1,
            window: int = 3
    ):
        """
        Initialize DateTimeSeriesSplit class with given n_splits, test_size, margin and window.

        Args:
            n_splits (int, optional): Number of folds. Defaults to 4.
            test_size (int, optional): Unique dates for out-of-fold sample.
            margin (int, optional): Number of margin in unique dates. Defaults to 1.
            window (int, optional): Number of unique dates as train. Defaults to 3.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.margin = margin
        self.window = window

    def get_n_splits(self) -> int:
        return self.n_splits

    def split(
            self,
            X: pd.DataFrame,
            y: Optional[Any] = None,
            groups: pd.DataFrame = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Args:
            X (pd.DataFrame): X dataset
            y (Optional[Any], optional): y dataset. Defaults to None.
            groups (pd.DataFrame, optional): column with date from X. Defaults to None.

        Yields:
            Iterator[Tuple[np.ndarray, np.ndarray]]: train and test ordinal number
        """
        unique_dates = sorted(groups.unique())
        rank_dates = {date: rank for rank, date in enumerate(unique_dates)}

        # Допишем вспомогательный столбец
        X['index_time'] = groups.map(rank_dates)
        X = X.reset_index(drop=True)
        index_time_list = list(rank_dates.values())

        for i in reversed(range(1, self.n_splits + 1)):
            left_train = int(
                (index_time_list[-1] - i * self.test_size + 1 - self.window - self.margin)
                * (self.window / np.max([1, self.window]))
            )
            right_train = index_time_list[-1] - i * self.test_size - self.margin + 1
            left_test = index_time_list[-1] - i * self.test_size + 1
            right_test = index_time_list[-1] - (i - 1) * self.test_size + 1

            index_test = X.index.get_indexer(
                X.index[X.index_time.isin(index_time_list[left_test:right_test])]
            )
            index_train = X.index.get_indexer(
                X.index[X.index_time.isin(index_time_list[left_train:right_train])]
            )

            yield index_train, index_test
