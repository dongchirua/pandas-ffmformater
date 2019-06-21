import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def generate_chunks(iterable, step=1):
    length = len(iterable)
    for ndx in range(0, length, step):
        yield iterable[ndx:min(ndx + step, length)]


class FFMformatter:
    def __init__(self, categorical_columns=None, numerical_columns=None, key_column='id', label_column='label', step=50):
        self.categorical = categorical_columns
        self.numerical = numerical_columns
        self.key_column = key_column
        self.label_column = label_column
        self.step = step

        self.result = {}
        self.field_count = -1
        self.feature_count = -1

        self.field_dict = defaultdict(lambda: self.__field_counter())
        self.feature_dict = defaultdict(lambda: defaultdict(lambda: self.__feature_counter()))

    def fit(self, df: pd.DataFrame, target=None):
        """
        df: Pandas DataFrame
        target: label column, str
        categorical: categorical columns, list
        numerical: numerical columns, list
        """

        self.__add_ids(df[self.key_column], target)

        ids = df[self.key_column]
        for field in df.columns:
            if field == self.key_column or field == self.label_column:
                continue
            for record_id, value in zip(ids, df[field]):

                if pd.isnull(value):
                    continue

                if field in self.categorical:
                    field_id = self.field_dict[field]
                    self.result[record_id] += " {}:{}:1".format(field_id,
                                                                self.feature_dict[field_id][value])
                elif field in self.numerical:
                    field_id = self.field_dict[field]
                    self.result[record_id] += " {}:{}:{}".format(field_id,
                                                                 field_id,
                                                                 value)
                else:
                    raise Exception(f"{field} was not defined in advance")

    def fit_transform(self, df: pd.DataFrame, target: np.ndarray = None):
        batches = list(generate_chunks(range(len(df)), self.step))
        for i in tqdm(batches, total=len(batches)):
            self.fit(df.iloc[i], target[i])
        return pd.Series(self.result).reset_index()

    def __add_ids(self, keys: np.ndarray, targets: np.ndarray = None):
        """
        This functions aims to init FFM data
        :param keys: objectID
        :param targets: label of this object
        """
        if targets is not None:
            for key, target in zip(keys, targets):
                self.result[key] = str(target)
        else:
            for key in keys:
                self.result[key] = "-1"

    def __field_counter(self):
        """
        Check https://github.com/ycjuan/libffm at Data Format
        """
        self.field_count += 1
        return self.field_count * 1

    def __feature_counter(self):
        """
        Check https://github.com/ycjuan/libffm at Data Format
        """
        self.feature_count += 1
        return self.feature_count * 1
