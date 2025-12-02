import os
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).parent.parent

# Model paths
MODEL_DIR = BASE_DIR / "models"
PLANTNET_MODEL_PATH = MODEL_DIR / "resnet152_weights_best_acc.tar"
PLANTNET_18_MODEL_PATH = MODEL_DIR / "resnet18_weights_best_acc.tar"
COMMON_NAMES_PATH = MODEL_DIR / "species_to_common_name.json"

# Server config
FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", False)

# Model config
DEVICE = "cpu"  # or "cuda" if GPU available
PLANTNET_INPUT_SIZE = 224
DISEASE_DETECTION_INPUT_SIZE = 224

# Request config
MAX_IMAGE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp"}
REQUEST_TIMEOUT_SEC = 60

# LLM config
LLM_MODEL_NAME = "mistral-7b"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 512