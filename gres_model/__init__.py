from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config, add_refcoco_config

# dataset loading
from .data.dataset_mappers.refcoco_mapper import RefCOCOMapper

# models
from .GRES import GRES

# evaluation
from .evaluation.refer_evaluation import ReferEvaluator
