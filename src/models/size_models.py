"""Paired initialization for size-conditioned GIN models."""
import random
import numpy as np
import torch
from src.models.gnn import GINCurvature, SizeFiLMGINCurvature

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def make_model(settings,markers,seed=None):
    kw=dict(n_markers=len(markers),hidden_dim=settings['HIDDEN_DIM'],num_layers=settings['NUM_LAYERS'],
        dropout=settings['DROPOUT'],residual=settings['RESIDUAL'],norm=settings['NORM'],global_dim=1)
    if seed is not None:seed_all(seed)
    model=SizeFiLMGINCurvature(**kw,film_hidden_dim=settings['FILM_HIDDEN_DIM'],size_feature_index=0)
    if seed is not None:
        seed_all(seed);normal=GINCurvature(**kw)
        # Exact common-weight initialization used by the original paired experiment.
        model.load_state_dict(normal.state_dict(),strict=False)
    return model
