import numpy as np

from environment.anomaly_detection.abstract_preprocessor import AbstractPreprocessor
from environment.settings import DROP_TEMPORAL, DUPLICATE_HEADERS, DROP_UNSTABLE


class AdvancedPreprocessor(AbstractPreprocessor):
    def __init__(self, correlation_threshold):
        self.const_feats = None
        self.correlated_feats = None
        self.threshold = correlation_threshold

    @staticmethod
    def get_constant_features(dataset):
        corr_const = dataset.corr()
        all_labels = set(corr_const.keys())
        # print("AD PRE: all", len(all_labels))

        corr_const.dropna(axis=1, how="all", inplace=True)
        corr_const.reset_index(drop=True)

        cropped_labels = set(corr_const.keys())
        # print("AD PRE: crop", len(cropped_labels))

        constant_feats = all_labels - cropped_labels
        # print("AD PRE: const", len(constant_feats))
        return constant_feats

    def get_highly_correlated_features(self, dataset):
        # https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
        corr_matrix = dataset.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # upper tri excl diagonal
        correlated_feats = [column for column in upper_tri.columns if any(upper_tri[column] > self.threshold)]
        # print("AD CORR:", len(correlated_feats))
        return correlated_feats

    def preprocess_dataset(self, dataset, is_normal=False):
        # Drop duplicated features
        dataset = dataset.drop(list(map(lambda header: header + ".1", DUPLICATE_HEADERS)), axis=1)  # read_csv adds the .1

        # Drop temporal features
        dataset = dataset.drop(DROP_TEMPORAL, axis=1)

        # Remove vectors generated when the rasp did not have connectivity
        if len(dataset) > 1:  # avoid dropping single entries causing empty dataset
            dataset = dataset.loc[dataset["connectivity"] == 1]

        # Drop unstable features
        dataset = dataset.drop(DROP_UNSTABLE, axis=1)

        # Drop constant features
        if self.const_feats is None:  # must preprocess normal data first to align infected data to it
            self.const_feats = AdvancedPreprocessor.get_constant_features(dataset)
        dataset = dataset.drop(self.const_feats, axis=1)

        # Drop highly correlated features
        if self.correlated_feats is None:  # must preprocess normal data first to align infected data to it
            self.correlated_feats = self.get_highly_correlated_features(dataset)
        dataset = dataset.drop(self.correlated_feats, axis=1)

        # Reset index
        dataset = dataset.reset_index(drop=True)
        return dataset
