from src.artifacts import pickle_compat as artifact_pickle
import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from src.artifacts.bundle import save_bundle, load_bundle, graph_membership, indexed_membership
from src.artifacts.checkpoints import build_model, model_spec, load_weights
from src.artifacts.paths import project_root
from src.artifacts.provenance import workflow_fingerprint
from src.models.gnn import GINCurvature, SizeFiLMGINCurvature
from src.data.target_transforms import StandardizeTransform


class TrainingArtifactsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_portable_named_handoff_preserves_predictions_and_fitted_transform(self):
        model = SizeFiLMGINCurvature(2, hidden_dim=8, num_layers=2, norm='batch', dropout=.1).eval()
        graph = Data(x=torch.eye(2), edge_index=torch.tensor([[0, 1], [1, 0]]),
                     y=torch.tensor([-.7, 1.2]), global_feat=torch.tensor([[.4]]), organoid_str='a')
        validation = copy.deepcopy(graph); validation.organoid_str = 'b'
        transform = StandardizeTransform().fit([graph])
        transformed = copy.deepcopy(validation)
        transform.transform_graphs([transformed], in_place=True)
        values = dict(models={'film': model}, alias=model, val=[transformed], transform=transform,
                      marker_names=['A', 'B'])
        original_x = graph.x.clone()
        out = self.root / 'run'
        save_bundle(out, values, splits=graph_membership(train=[graph], validation=[validation]))
        restored = load_bundle(out)
        self.assertIs(restored['models']['film'], restored['alias'])
        batch = Batch.from_data_list([transformed])
        loaded_batch = Batch.from_data_list(restored['val'])
        with torch.no_grad():
            expected, hidden = model(batch.x, batch.edge_index, data=batch)
            actual, loaded_hidden = restored['alias'](loaded_batch.x, loaded_batch.edge_index, data=loaded_batch)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(loaded_hidden, hidden, rtol=0, atol=0)
        np.testing.assert_array_equal(restored['transform'].inverse(transformed.y), transform.inverse(transformed.y))
        torch.testing.assert_close(graph.x, original_x)
        self.assertFalse(restored['alias'].training)
        with self.assertRaises(FileExistsError):
            save_bundle(out, values, splits={})
        with (out / 'inputs.pkl').open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            load_bundle(out)

    def test_all_historical_checkpoint_formats_and_spec_roundtrip(self):
        original = GINCurvature(2, hidden_dim=8, num_layers=1, global_dim=1, dropout=.37, norm='layer')
        spec = json.loads(json.dumps(model_spec(original)))
        self.assertEqual(spec['kwargs']['dropout'], .37)
        for index, payload in enumerate([original.state_dict(), {'model_state_dict': original.state_dict()},
                                         {'model_state': original.state_dict()}]):
            path = self.root / f'{index}.pt'; torch.save(payload, path)
            restored = load_weights(build_model(spec), path)
            for name, expected in original.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)

    def test_fold_membership_is_explicit_and_disjoint(self):
        graphs = [Data(organoid_str=f'org_{i}') for i in range(3)]
        actual = indexed_membership(graphs, [dict(fold=0, train_indices=[2, 0], val_indices=[1])])
        self.assertEqual(actual[0]['train'], ['org_2', 'org_0'])
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            graph_membership(train=graphs[:2], validation=graphs[1:])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            graph_membership(train=[graphs[0], graphs[0]])

    def test_legacy_pickle_class_imports_still_resolve(self):
        from src.analysis.embeddings.response_geometry import WeightedPCA
        from src.analysis.embeddings.size_responses import EmbeddingConfig
        from src.training.masking import MaskTrainingConfig
        self.assertIs(artifact_pickle.loads(b'csrc.analysis.ki67_pca\nWeightedPCA\n.'), WeightedPCA)
        self.assertIs(artifact_pickle.loads(b'csrc.analysis.fate_masking\nMaskTrainingConfig\n.'), MaskTrainingConfig)
        self.assertIs(artifact_pickle.loads(b'csrc.analysis.size_embedding\nEmbeddingConfig\n.'), EmbeddingConfig)
        self.assertIs(artifact_pickle.loads(b'csrc.analysis.embeddings.size_conditioning\nEmbeddingConfig\n.'), EmbeddingConfig)

    def test_workflow_hash_ignores_outputs_but_tracks_code_and_helpers(self):
        (self.root / 'src/models').mkdir(parents=True)
        helper = self.root / 'src/models/gnn.py'; helper.write_text('value = 1\n')
        notebook = self.root / 'experiments' / 'nested' / 'analysis.ipynb'
        notebook.parent.mkdir(parents=True)
        obj = {'cells': [dict(cell_type='code', source=['x = 1\n'], outputs=[], execution_count=None)]}
        notebook.write_text(json.dumps(obj)); original = workflow_fingerprint(notebook)
        obj['cells'][0].update(outputs=[{'text': 'a new figure'}], execution_count=8)
        notebook.write_text(json.dumps(obj)); self.assertEqual(original, workflow_fingerprint(notebook))
        helper.write_text('value = 2\n'); self.assertNotEqual(original, workflow_fingerprint(notebook))
        self.assertEqual(project_root(notebook.parent), self.root)

    def test_saved_cohort_selection_ignores_extra_source_graphs_and_keeps_order(self):
        from src.analysis.size_conditioning.cohort_inputs import load_cohort
        import pandas as pd
        graphs = [Data(x=torch.eye(2), y=torch.tensor([0., 1.]),
                       edge_index=torch.tensor([[0, 1], [1, 0]]), organoid_str=name)
                  for name in ['excluded', 'b', 'a']]
        (self.root / 'tables').mkdir()
        pd.DataFrame({'organoid_str': ['a', 'b']}).to_csv(self.root / 'tables/cohort.csv', index=False)
        module = 'src.analysis.size_conditioning.cohort_inputs'
        with patch(module + '.load_graph_dataset_from_dir', return_value=graphs), \
             patch(module + '.load_aux_metadata_for_dir', return_value={}), \
             patch(module + '.attach_metadata_to_graphs'):
            selected = load_cohort(self.root, {'INTERPOLATE_TARGET_OUTLIERS': False}, self.root)
        self.assertEqual([g.organoid_str for g in selected], ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
