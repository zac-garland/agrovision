import os
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).parent.parent

# Model paths
MODEL_DIR = BASE_DIR / "models"
PLANTNET_MODEL_PATH = MODEL_DIR / "efficientnet_b4_weights_best_acc.tar"  # General plant model (PlantNet-300K)
PLANTNET_18_MODEL_PATH = MODEL_DIR / "resnet18_weights_best_acc.tar"  # Keep for backward compatibility
EFFICIENTNET_MODEL_PATH = MODEL_DIR / "efficientnet_b4_weights_best_acc.tar"
HOUSEPLANT_MODEL_PATH = MODEL_DIR / "efficientnet_b4_houseplant_finetuned.tar"  # Fine-tuned houseplant model
SPECIES_CLASSIFIER_PATH = MODEL_DIR / "efficientnet_b4_species_classifier.tar"  # Unified species classifier
SPECIES_CLASSIFIER_DEV_PATH = MODEL_DIR / "efficientnet_b4_species_classifier_dev.tar"  # Dev version
DISEASE_CLASSIFIER_PATH = MODEL_DIR / "efficientnet_b4_disease_classifier.tar"  # Disease classifier
DISEASE_CLASSIFIER_DEV_PATH = MODEL_DIR / "efficientnet_b4_disease_classifier_dev.tar"  # Dev version
COMMON_NAMES_PATH = MODEL_DIR / "species_to_common_name.json"
YOLO_LEAF_MODEL_PATH = MODEL_DIR / "yolo11x_leaf.pt"

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
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "mistral")  # Ollama model name
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))  # Increased to prevent text cutoff
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")