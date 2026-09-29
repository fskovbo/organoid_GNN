"""Read trusted local scientific artifacts after module moves.

Only historical pickle class paths are remapped; ordinary imports use canonical
modules. This is compatibility handling, not a safe loader for untrusted files.
"""
import pickle
import io

MODULE_MOVES = {
    'src.models.fate_spline': 'legacy.fate_interactions.models.fate_spline',
    'src.models.fate_fraction': 'legacy.fate_interactions.models.fate_fraction',
    'src.training.spline_fit': 'legacy.fate_interactions.training.spline_fit',

    'src.analysis.cluster_analysis': 'src.analysis.embeddings.clusters',
    'src.analysis.motif_clustering': 'src.analysis.embeddings.clustering',
    'src.analysis.size_embedding': 'src.analysis.embeddings.size_responses',
    'src.analysis.size_embedding_focus': 'src.analysis.embeddings.readout',
    'src.analysis.ki67_pca': 'src.analysis.embeddings.response_geometry',
    'src.analysis.perturbation': 'src.analysis.interventions.perturbation',
    'src.analysis.pseudotime': 'src.analysis.interventions.size_sweeps',
    'src.analysis.replacement_ablation': 'src.analysis.interventions.replacement',
    'src.analysis.masking_ablation': 'src.analysis.interventions.masking_comparison',
    'src.analysis.fate_masking': 'src.analysis.interventions.masking',
    'src.analysis.exclusive_size_ablation': 'src.analysis.size_conditioning.cohort_inputs',
    'src.analysis.exclusive_comparison': 'src.analysis.size_conditioning.encoding_effects',
    'src.analysis.exclusive_pair_control': 'src.analysis.size_conditioning.pair_control',
    'src.analysis.lgr5_film': 'src.analysis.conditioning.film',
    'src.analysis.lgr5_film_summary': 'src.analysis.conditioning.response_statistics',
    'src.analysis.geometric_normalization': 'src.analysis.normalization.geometry',
    'src.analysis.prediction_analysis': 'src.analysis.metrics.prediction',
    'src.analysis.marker_stats': 'src.analysis.metrics.markers',
    'src.analysis.composition': 'src.analysis.metrics.composition',
    'src.analysis.ki67_observed': 'src.analysis.spatial.neighborhoods',
    'src.analysis.ki67_neck_validation': 'src.analysis.spatial.necks',
    'src.analysis.unassigned_cells': 'src.analysis.spatial.unassigned',
    'src.analysis.marker_domain': 'src.analysis.spatial.domains',
    'src.analysis.niche_hypotheses': 'src.analysis.spatial.niche_inference',
    'src.analysis.size_conditioning.data': 'src.analysis.size_conditioning.cohort_inputs',
    'src.analysis.size_conditioning.comparison': 'src.analysis.size_conditioning.encoding_effects',
    'src.analysis.spatial.niche': 'src.analysis.spatial.niche_inference',
    'src.analysis.conditioning.summary': 'src.analysis.conditioning.response_statistics',
    'src.models.size_conditioning': 'src.models.size_models',
    'src.analysis.embeddings.size_conditioning': 'src.analysis.embeddings.size_responses',
}


class ArtifactUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(MODULE_MOVES.get(module, module), name)


def load(handle):
    return ArtifactUnpickler(handle).load()


def loads(data):
    return load(io.BytesIO(data))
