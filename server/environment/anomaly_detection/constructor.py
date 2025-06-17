from pyod.models.iforest import IForest
from pyod.models.lof import LOF
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
        if proto in []:
            PREPROCESSOR = SimplePreprocessor()
        elif proto in ["8", "20", "21", "24"]:
            PREPROCESSOR = AdvancedPreprocessor(MAX_ALLOWED_CORRELATION_IF)
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
        if proto in ["8", "20", "21", "24"]:
            # CLASSIFIER = IForest(random_state=42, contamination=CONTAMINATION_FACTOR)
            # CLASSIFIER = LOF(n_neighbors=1500, contamination=CONTAMINATION_FACTOR)
            CLASSIFIER = OneClassSVM(kernel="rbf", gamma="auto", nu=CONTAMINATION_FACTOR)
        else:
            print("WARNING: Unknown prototype. Falling back to Isolation Forest classifier!")
            CLASSIFIER = IForest(random_state=42, contamination=CONTAMINATION_FACTOR)
    return CLASSIFIER
