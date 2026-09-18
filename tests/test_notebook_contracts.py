"""Cheap structural checks; do not train models or touch scientific outputs."""
import ast
import json
import os
import unittest
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


class NotebookContractTests(unittest.TestCase):
    def test_layout_introductions_syntax_and_analysis_training_boundary(self):
        self.assertFalse(list((ROOT / 'experiments').glob('*.ipynb')))
        forbidden = {'train', 'train_mask_model', 'train_masking_experiment', 'train_or_load_model',
                     'train_model_grid_for_variant_packs', 'prepare_fold_graphs', 'prepare_training_graphs'}
        for path in (ROOT / 'experiments').rglob('*.ipynb'):
            with self.subTest(notebook=str(path.relative_to(ROOT))):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                self.assertEqual(notebook.cells[0].cell_type, 'markdown')
                self.assertIn('Role:', notebook.cells[0].source)
                for cell in notebook.cells:
                    if cell.cell_type != 'code':
                        continue
                    tree = ast.parse(cell.source)
                    if 'training' not in path.parts and 'archive' not in path.parts:
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                                self.assertNotIn(node.func.id, forbidden)

    def test_consolidated_training_and_no_compatibility_shims(self):
        expected = {'gin_depth_training.ipynb',
                    'fate_masking_training.ipynb', 'graph_controls_training.ipynb',
                    'lineage_removal_training.ipynb'}
        self.assertEqual({p.name for p in (ROOT/'experiments/training').glob('*.ipynb')}, expected)
        for obsolete in ('analysis', 'archive', 'figures'):
            self.assertFalse((ROOT/'experiments'/obsolete).exists())
        self.assertFalse(list((ROOT/'legacy').rglob('*.ipynb')))
        for path in (ROOT/'src/analysis').glob('*.py'):
            self.assertNotIn('sys.modules[__name__]', path.read_text())
        for path in (ROOT/'experiments').rglob('*.ipynb'):
            notebook = nbformat.read(path, as_version=4)
            self.assertGreaterEqual(len(notebook.cells[0].source.split()), 45, str(path))
            source = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
            self.assertNotIn('globals().update(', source)

    def test_bootstrap_works_from_every_notebook_directory(self):
        before = Path.cwd()
        try:
            for path in (ROOT / 'experiments').rglob('*.ipynb'):
                notebook = json.loads(path.read_text())
                os.chdir(path.parent)
                namespace = {}
                exec(''.join(notebook['cells'][1]['source']), namespace)
                self.assertEqual(namespace['PROJECT_ROOT'], ROOT)
        finally:
            os.chdir(before)

    def test_migration_targets_exist(self):
        mapping = json.loads((ROOT / 'docs/notebook_migration.json').read_text())
        for targets in mapping.values():
            for target in targets:
                self.assertTrue((ROOT / 'experiments' / target).is_file(), target)


if __name__ == '__main__':
    unittest.main()
