"""Checkpoint-based diagnostics of size-conditioned marker responses at LGR5 centers.

Inference only: split FiLM/head conditioning, trace hidden perturbations, and
compute factorial finite differences on fixed held-out neighborhoods.
"""
import copy
import itertools
import json
import pickle
import hashlib
from pathlib import Path
from contextlib import contextmanager
from types import MethodType

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from src.data.io import load_organoid_npz, build_pyg_graph
from src.data.subgraphs import build_ego_subgraphs_for_graph
from src.models.gnn import SizeFiLMGINCurvature

ROUTES = ['full', 'film_only', 'head_only', 'layer1_only', 'layer2_only']


@contextmanager
def independent_conditioning(model, film_sizes):
    """Temporarily supply one standardized size per FiLM layer; head uses data globals."""
    original = model._condition_update
    def condition(self, h, layer_index, data):
        size = h.new_full((data.num_graphs, 1), float(film_sizes[layer_index]))
        gamma, beta = self.film_layers[layer_index](size).chunk(2, dim=-1)
        return (1 + gamma[data.batch]) * h + beta[data.batch]
    model._condition_update = MethodType(condition, model)
    try:
        yield
    finally:
        model._condition_update = original


@torch.no_grad()
def infer_requests(model, subgraphs, requests, *, head_size, film_sizes, device, batch_size=256):
    """Run actual model forward, capturing post-residual hidden states before the head."""
    means, layer1, layer2 = [], [], []
    model.to(device).eval()
    with independent_conditioning(model, film_sizes):
        for start in range(0, len(requests), batch_size):
            items = []
            for si, edits in requests[start:start + batch_size]:
                graph = copy.copy(subgraphs[si])
                graph.x = graph.x.clone()
                for node, marker in edits:
                    graph.x[node, marker] = 0
                graph.global_feat = graph.x.new_tensor([[head_size]])
                items.append(graph)
            batch = Batch.from_data_list(items).to(device)
            centers = batch.ptr[:-1] + batch.center_idx
            captured = {}
            def capture_first(_module, inputs):
                captured['h1'] = inputs[0][centers].detach().cpu().numpy()
            def capture_last(_module, inputs):
                captured['h2'] = inputs[0][centers, :model.hidden_dim].detach().cpu().numpy()
            hooks = [model.convs[1].register_forward_pre_hook(capture_first),
                     model.head.register_forward_pre_hook(capture_last)]
            try:
                (mu, _), _ = model(batch.x, batch.edge_index, batch)
            finally:
                for hook in hooks:
                    hook.remove()
            means.append(mu[centers].reshape(-1).cpu().numpy())
            layer1.append(captured['h1'])
            layer2.append(captured['h2'])
    return dict(z=np.concatenate(means).astype(float), h1=np.concatenate(layer1), h2=np.concatenate(layer2))


def hidden_comparison(delta, reference, floor=1e-8):
    """Magnitude and within-model direction; undefined for zero reference/effect."""
    norm, refnorm = np.linalg.norm(delta, axis=1), np.linalg.norm(reference, axis=1)
    ratio = np.divide(norm, refnorm, out=np.full_like(norm, np.nan), where=refnorm > floor)
    cosine = np.divide(np.sum(delta * reference, axis=1), norm * refnorm,
                       out=np.full_like(norm, np.nan), where=(norm > floor) & (refnorm > floor))
    return norm, ratio, np.clip(cosine, -1, 1)


def mixed_difference(values, indices):
    base, a, b, both = np.asarray(indices).T
    return values[both] - values[a] - values[b] + values[base]


def build_requests(subgraphs, cases, lgr5_index):
    """Deduplicate perturbations; neighbor pairs use distinct cells in the same hop."""
    requests, lookup = [], {}
    def request(si, edits=()):
        key = (si, tuple(sorted(set(edits))))
        if key not in lookup:
            lookup[key] = len(requests)
            requests.append(key)
        return lookup[key]
    singles = []
    for case in cases:
        si = case['subgraph_index']
        singles.append((request(si), request(si, [(case['source_node'], case['source_marker'])])))
    single_request_count = len(requests)
    interactions, indices = [], []
    for case, (base, single) in zip(cases, singles):
        si = case['subgraph_index']
        center_edit = (int(subgraphs[si].center_idx), lgr5_index)
        source_edit = (case['source_node'], case['source_marker'])
        indices.append((base, request(si, [center_edit]), single, request(si, [center_edit, source_edit])))
        interactions.append(dict(kind='center_neighbor', marker_a='center LGR5',
                                 marker_b=case['source_marker_name'], hop=case['hop'],
                                 organoid_str=case['organoid_str'], orig_center=case['orig_center'],
                                 source_a=int(subgraphs[si].orig_center), source_b=case['orig_source_node'],
                                 center_markers='|'.join(case['center_marker_names'])))
    grouped = {}
    for case in cases:
        grouped.setdefault((case['subgraph_index'], case['hop']), []).append(case)
    skipped = 0
    for (si, hop), group in grouped.items():
        for a, b in itertools.combinations(sorted(group, key=lambda c: c['source_marker_name']), 2):
            if a['source_node'] == b['source_node']:
                skipped += 1
                continue
            ae, be = (a['source_node'], a['source_marker']), (b['source_node'], b['source_marker'])
            indices.append((request(si), request(si, [ae]), request(si, [be]), request(si, [ae, be])))
            interactions.append(dict(kind='neighbor_neighbor', marker_a=a['source_marker_name'],
                                     marker_b=b['source_marker_name'], hop=hop,
                                     organoid_str=a['organoid_str'], orig_center=a['orig_center'],
                                     source_a=a['orig_source_node'], source_b=b['orig_source_node'],
                                     center_markers='|'.join(a['center_marker_names'])))
    return requests, np.asarray(singles), single_request_count, pd.DataFrame(interactions), np.asarray(indices), skipped


def prepare_fold(run_dir, data_dir, fold, lgr5_index):
    manifest = json.loads((run_dir / f'tables/fold_{fold}_sweep_manifest.json').read_text())
    cases = [dict(c) for c in manifest if 'LGR5' in c['center_marker_names']]
    membership = pd.read_csv(run_dir / 'tables/split_membership.csv')
    allowed = set(membership[(membership.fold == fold) & (membership.role == 'val')].organoid_str)
    assert {c['organoid_str'] for c in cases} <= allowed
    subgraphs, lookup = [], {}
    for organoid in sorted({c['organoid_str'] for c in cases}):
        arr = load_organoid_npz(str(data_dir / f'{organoid}.npz'), strict=True, target_indices=[0])
        graph = build_pyg_graph(arr['x'], arr['edges'], arr['y'])
        graph.organoid_str = organoid
        centers = sorted({c['orig_center'] for c in cases if c['organoid_str'] == organoid})
        for sub in build_ego_subgraphs_for_graph(graph, num_hops=2, centers=centers):
            assert sub.x[int(sub.center_idx), lgr5_index] > .5
            lookup[(organoid, int(sub.orig_center))] = len(subgraphs)
            subgraphs.append(sub)
    for case in cases:
        case['subgraph_index'] = lookup[(case['organoid_str'], case['orig_center'])]
        sub = subgraphs[case['subgraph_index']]
        location = torch.nonzero(sub.orig_nodes == case['orig_source_node']).item()
        # Same extraction and source selection as the saved experiment.
        assert location == case['source_node']
        assert sub.x[location, case['source_marker']] > .5
    with (run_dir / f'checkpoints/fold_{fold}_preprocessing.pkl').open('rb') as handle:
        preprocessing = pickle.load(handle)
    return subgraphs, cases, preprocessing


def run_diagnostics(run_dir, project_root, *, device='cuda', batch_size=256, output_name='lgr5_film_diagnostics'):
    """Execute all five routes, layer traces and factorial ablations; resume by checkpoint."""
    run_dir, project_root = Path(run_dir), Path(project_root)
    settings = json.loads((run_dir / 'settings.json').read_text())
    data_dir = project_root / 'training_data' / settings['DATASET_NAME']
    marker_names = json.loads(next(data_dir.glob('*_markers.json')).read_text())
    lgr5_index = marker_names.index('LGR5')
    counts = json.loads((run_dir / 'sweep_grid.json').read_text())
    reference_n = counts[len(counts) // 2]
    references = pd.read_csv(run_dir / 'geometric_normalization/references.csv').set_index('fold')
    out = run_dir / output_name
    out.mkdir(exist_ok=True)
    config = dict(version=1, run_dir=str(run_dir), device=device, batch_size=batch_size,
                  model='gin_film_size', counts=counts, reference_n=reference_n,
                  routes=ROUTES, seeds=settings['MODEL_SEEDS'], marker_names=marker_names,
                  sample='all originally LGR5-positive centers in saved sweep manifests',
                  neighbor_pairs='same hop, different markers, distinct saved source cells')
    config_path = out / 'settings.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        assert all(previous[k] == config[k] for k in config if k not in ('device', 'batch_size')), 'Incompatible existing diagnostics.'
    config_path.write_text(json.dumps(config, indent=2))
    torch.set_num_threads(4)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use an environment with GPU access or device="cpu".')
    support, validation = [], []
    for fold in references.index:
        print(f'Prepare fold {fold}', flush=True)
        subgraphs, cases, preprocessing = prepare_fold(run_dir, data_dir, fold, lgr5_index)
        requests, singles, n_single, interaction_meta, interaction_indices, skipped = build_requests(subgraphs, cases, lgr5_index)
        pd.DataFrame(cases).to_json(out / f'fold_{fold}_single_manifest.json', orient='records', indent=2)
        interaction_meta.to_csv(out / f'fold_{fold}_interaction_manifest.csv', index=False)
        support.append(dict(fold=fold, centers=len(subgraphs), organoids=len({c['organoid_str'] for c in cases}),
                            single_cases=len(cases), neighbor_pairs=int((interaction_meta.kind == 'neighbor_neighbor').sum()),
                            skipped_same_source_pairs=skipped))
        meta = pd.DataFrame([{k: c[k] for k in ['case_id', 'organoid_str', 'orig_center', 'source_marker_name', 'hop']}
                             | {'center_markers': '|'.join(c['center_marker_names'])} for c in cases])
        transform = preprocessing['residual_transform']
        def standardized(n):
            return (np.log(n) - preprocessing['size_center']) / preprocessing['size_scale']
        def multiplier(n):
            r = references.loc[fold]
            return np.exp(r.alpha + r.beta * np.log(n)) / (4 * np.pi)
        ref_t = standardized(reference_n)
        for seed in settings['MODEL_SEEDS']:
            prefix = out / f'fold_{fold}_seed_{seed}'
            done = Path(str(prefix) + '_done.json')
            if done.exists():
                validation.append(json.loads(done.read_text()))
                print(f'Already complete: fold {fold}, seed {seed}', flush=True)
                continue
            model = SizeFiLMGINCurvature(len(marker_names), hidden_dim=settings['HIDDEN_DIM'],
                num_layers=settings['NUM_LAYERS'], global_dim=1, film_hidden_dim=settings['FILM_HIDDEN_DIM'],
                dropout=settings['DROPOUT'], norm=settings['NORM'], residual=settings['RESIDUAL'])
            checkpoint = run_dir / f'checkpoints/fold_{fold}_seed_{seed}_gin_film_size.pt'
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            ref_pred = infer_requests(model, subgraphs, requests[:n_single], head_size=ref_t,
                                     film_sizes=[ref_t, ref_t], device=device, batch_size=batch_size)
            ref_hidden = {key: ref_pred[key][singles[:, 1]] - ref_pred[key][singles[:, 0]] for key in ['h1', 'h2']}
            # Compare instrumented forward directly with original forward at the reference.
            check_graphs = []
            for graph in subgraphs[:8]:
                g = copy.copy(graph); g.global_feat = g.x.new_tensor([[ref_t]]); check_graphs.append(g)
            batch = Batch.from_data_list(check_graphs).to(device)
            with torch.no_grad():
                direct, _ = model(batch.x, batch.edge_index, batch)
            direct_z = direct[0][batch.ptr[:-1] + batch.center_idx].reshape(-1).cpu().numpy()
            lookup = {req: i for i, req in enumerate(requests)}
            instrument_z = ref_pred['z'][[lookup[(i, ())] for i in range(len(check_graphs))]]
            np.testing.assert_allclose(instrument_z, direct_z, atol=2e-6, rtol=2e-5)
            effects, interactions, max_old_difference = [], [], 0.
            for n in counts:
                t = standardized(n)
                for route in ROUTES:
                    head = ref_t if route in ('film_only', 'layer1_only', 'layer2_only') else t
                    film = {'full': [t, t], 'film_only': [t, t], 'head_only': [ref_t, ref_t],
                            'layer1_only': [t, ref_t], 'layer2_only': [ref_t, t]}[route]
                    all_requests = requests if route == 'full' else requests[:n_single]
                    pred = infer_requests(model, subgraphs, all_requests, head_size=head, film_sizes=film,
                                          device=device, batch_size=batch_size)
                    raw = np.asarray(transform.inverse(pred['z'])).reshape(-1)
                    table = meta.assign(fold=fold, seed=seed, n=n, route=route,
                        delta_z=pred['z'][singles[:, 1]] - pred['z'][singles[:, 0]],
                        delta_raw=raw[singles[:, 1]] - raw[singles[:, 0]])
                    table['delta_relative'] = table.delta_raw * multiplier(n)
                    table['delta_fixed_reference'] = table.delta_raw * multiplier(reference_n)
                    for key in ['h1', 'h2']:
                        delta = pred[key][singles[:, 1]] - pred[key][singles[:, 0]]
                        norm, ratio, cosine = hidden_comparison(delta, ref_hidden[key])
                        table[f'{key}_norm'], table[f'{key}_ratio'], table[f'{key}_cosine'] = norm, ratio, cosine
                    if route == 'head_only':
                        for key in ['h1', 'h2']:
                            np.testing.assert_allclose(pred[key][:n_single], ref_pred[key], atol=1e-5, rtol=5e-5)
                    effects.append(table)
                    if route == 'full':
                        old = pd.read_csv(run_dir / f'tables/fold_{fold}_seed_{seed}_gin_film_size_N{n}_sweep.csv',
                                          usecols=['case_id', 'delta_mu'])
                        matched = table.merge(old, on='case_id', validate='one_to_one')
                        assert len(matched) == len(cases)
                        max_old_difference = max(max_old_difference, float(np.max(np.abs(matched.delta_raw - matched.delta_mu))))
                        np.testing.assert_allclose(matched.delta_raw, matched.delta_mu, atol=2e-6, rtol=2e-4)
                        it = interaction_meta.assign(fold=fold, seed=seed, n=n,
                            interaction_z=mixed_difference(pred['z'], interaction_indices),
                            interaction_raw=mixed_difference(raw, interaction_indices))
                        it['interaction_relative'] = it.interaction_raw * multiplier(n)
                        for key in ['h1', 'h2']:
                            delta = mixed_difference(pred[key], interaction_indices)
                            base, a, b, both = interaction_indices.T
                            denom = np.linalg.norm(pred[key][a] - pred[key][base], axis=1) + np.linalg.norm(pred[key][b] - pred[key][base], axis=1)
                            it[f'{key}_interaction_norm'] = np.linalg.norm(delta, axis=1)
                            it[f'{key}_interaction_ratio'] = np.divide(np.linalg.norm(delta, axis=1), denom,
                                out=np.full(len(denom), np.nan), where=denom > 1e-8)
                        interactions.append(it)
                print(f'fold={fold}, seed={seed}, N={n}: five routes + factorial ablations', flush=True)
            pd.concat(effects, ignore_index=True).to_csv(str(prefix) + '_effects.csv.gz', index=False, compression='gzip')
            pd.concat(interactions, ignore_index=True).to_csv(str(prefix) + '_interactions.csv.gz', index=False, compression='gzip')
            record = dict(fold=int(fold), seed=int(seed), max_saved_raw_difference=max_old_difference,
                          checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            done.write_text(json.dumps(record, indent=2)); validation.append(record)
            print(f'Saved fold={fold}, seed={seed}; max reproduction error {max_old_difference:.3g}', flush=True)
            model.cpu(); del model
            if device == 'cuda': torch.cuda.empty_cache()
    pd.DataFrame(support).to_csv(out / 'sampling_support.csv', index=False)
    pd.DataFrame(validation).to_csv(out / 'reproduction_checks.csv', index=False)
    return out
