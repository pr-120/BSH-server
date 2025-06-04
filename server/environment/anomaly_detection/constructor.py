from pyod.models.iforest import IForest
from sklearn.svm import OneClassSVM

from environment.anomaly_detection.advanced_preprocessor import AdvancedPreprocessor
from environment.anomaly_detection.autoencoder import AutoEncoder
from environment.anomaly_detection.simple_preprocessor import SimplePreprocessor
from environment.settings import MAX_ALLOWED_CORRELATION_IF, MAX_ALLOWED_CORRELATION_AE
from environment.state_handling import get_prototype

# ========================================
# ==========   CONFIG   ==========
# ========================================
CONTAMINATION_FACTOR = 0.01

# ========================================
# ==========   GLOBALS   ==========
# ========================================
CLASSIFIER = None
PREPROCESSOR = None


def get_preprocessor():
    global PREPROCESSOR
    if not PREPROCESSOR:
        proto = get_prototype()
        if proto in ["1", "2", "99"]:
            PREPROCESSOR = SimplePreprocessor()
        elif proto in ["3", "4", "5", "6", "7", "8", "20", "21", "22", "23", "24", "25", "20", "21", "22", "23", "24",
                       "25"]:
            PREPROCESSOR = AdvancedPreprocessor(__get_correlation_threshold())
        else:
            print("WARNING: Unknown prototype. Falling back to default simple preprocessor!")
            PREPROCESSOR = SimplePreprocessor()
    return PREPROCESSOR


def reset_AD():
    global PREPROCESSOR
    PREPROCESSOR = None
    global CLASSIFIER
    CLASSIFIER = None


def get_classifier():
    global CLASSIFIER
    if not CLASSIFIER:
        proto = get_prototype()
        if proto in ["1", "2", "3", "4", "5", "6", "7", "8", "10", "98", "99", "20", "21", "22", "23", "24", "25"]:
            # CLASSIFIER = IForest(random_state=42, contamination=CONTAMINATION_FACTOR)
            # CLASSIFIER = LOF(n_neighbors=1500, contamination=CONTAMINATION_FACTOR)
            CLASSIFIER = OneClassSVM(kernel="rbf", gamma="auto", nu=CONTAMINATION_FACTOR)
        elif proto in ["9"]:
            CLASSIFIER = AutoEncoder(encoding_dim=[40, 20, 10, 20, 40], random_state=42,
                                     outlier_percentage=CONTAMINATION_FACTOR)
        else:
            print("WARNING: Unknown prototype. Falling back to Isolation Forest classifier!")
            CLASSIFIER = IForest(random_state=42, contamination=CONTAMINATION_FACTOR)
    return CLASSIFIER


def __get_correlation_threshold():
    proto = get_prototype()
    if proto in ["9"]:
        return MAX_ALLOWED_CORRELATION_AE
    else:
        return MAX_ALLOWED_CORRELATION_IF
