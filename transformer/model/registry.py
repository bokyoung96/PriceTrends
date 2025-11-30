from core.models.model1 import CNNModel
from transformer.model.model1 import Transformer
from transformer.model.model2 import MultiHeadTransformer
from transformer.model.model3 import CrashTransformer, MultiModalCrash

MODEL_REGISTRY = {
    "transformer": Transformer,
    "multi": MultiHeadTransformer,
    "crash": CrashTransformer,
    "cnn": CNNModel,
    "multimodal_crash": MultiModalCrash,
}

__all__ = ["MODEL_REGISTRY"]
