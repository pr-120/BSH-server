from environment.anomaly_detection.abstract_preprocessor import AbstractPreprocessor
from environment.settings import DROP_CONNECTIVITY, DROP_CONSTANT, DROP_TEMPORAL, DUPLICATE_HEADERS


class SimplePreprocessor(AbstractPreprocessor):
    def preprocess_dataset(self, dataset):
        # Remove vectors generated when the rasp did not have connectivity
        if len(dataset) > 1:  # avoid dropping single entries causing empty dataset
            dataset = dataset.loc[dataset["connectivity"] == 1].copy()
        # Remove the connectivity feature because now it is constant
        dataset = dataset.drop(DROP_CONNECTIVITY, axis=1)

        # Remove duplicates
        dataset = dataset.drop(list(map(lambda header: header + ".1", DUPLICATE_HEADERS)),
                     axis=1)  # read_csv adds the .1

        # Remove temporal features
        dataset = dataset.drop(DROP_TEMPORAL, axis=1)

        # Remove constant features
        dataset = dataset.drop(DROP_CONSTANT, axis=1)

        # Reset index
        dataset = dataset.reset_index(drop=True)
        return dataset
