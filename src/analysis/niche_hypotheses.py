"""Held-out, recipient-preserving niche diagnostics. No training is performed.

Full graphs preserve every recipient's receptive field. A marker ablation clears
one binary feature, never removes a cell or changes the supplied cell count.
"""
from pathlib import Path
import copy
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from src.data.io import load_organoid_npz, build_pyg_graph
from src.models.gnn import SizeFiLMGINCurvature

MARKERS = ['Agr2', 'AldoB', 'Chroma', 'KI67', 'LGR5', 'Lysozyme', 'Serotonin']
FAMILIES = [('Agr2', 'Lysozyme'), ('Chroma', 'Serotonin')]


def rings(adjacency, node):
    one = set(adjacency[node]) - {node}
    two = set().union(*(adjacency[j] for j in one)) - one - {node} if one else set()
    return [sorted(one), sorted(two)]


def source_state(x, precursor, mature):
    a, b = x[:, MARKERS.index(precursor)] > .5, x[:, MARKERS.index(mature)] > .5
    return np.select([a & ~b, a & b, ~a & b], ['precursor_only', 'double_positive', 'mature_only'], default='neither')


def crypt_membership(projection, crypts):
    labels = np.full(len(projection), -1, dtype=int)
    for k, vertices in enumerate(crypts):
        selected = np.isin(projection, np.asarray(vertices, dtype=int))
        if np.any(labels[selected] >= 0):
            raise ValueError('Overlapping final crypt labels')
        labels[selected] = k
    return labels


def prepare(run_dir, data_dir, out, cap=3, sampling_seed=2301):
    """Outcome-independent sampling fixed across model seeds and supplied N."""
    membership = pd.read_csv(run_dir / 'tables/split_membership.csv')
    membership = membership[membership.role == 'val'].sort_values(['fold', 'organoid_str'])
    assert not membership.organoid_str.duplicated().any()
    rng = np.random.default_rng(sampling_seed)
    graphs, requests, singles, pairs, census = {}, {}, [], [], []
    for row in membership.itertuples():
        org, fold = row.organoid_str, int(row.fold)
        arr = load_organoid_npz(str(data_dir / f'{org}.npz'), strict=True, target_indices=[0])
        graph = build_pyg_graph(arr['x'], arr['edges'], arr['y'])
        graph.organoid_str = org
        x = graph.x.numpy(); n = len(x)
        meta = json.loads((data_dir / f'{org}_aux.json').read_text())
        with np.load(data_dir / f'{org}.npz') as f:
            projection = f['proj_vertex_ids']; curvature = f['y'][:, 0]
        # Locally generated trusted object array of mesh-vertex index lists.
        with np.load(meta['segmentation_path'], allow_pickle=True) as f:
            crypts = f['crypts_ll']; labels = crypt_membership(projection, crypts)
        regions = np.where(labels >= 0, 'detected_crypt', 'outside_detected_crypt' if len(crypts) else 'no_crypt_detected')
        adjacency = [set() for _ in range(n)]
        for a, b in graph.edge_index.numpy().T:
            adjacency[a].add(int(b)); adjacency[b].add(int(a))
        neighborhoods = [rings(adjacency, i) for i in range(n)]
        graphs.setdefault(fold, []).append(graph)
        gi = len(graphs[fold]) - 1
        req = requests.setdefault(fold, [])
        lookup = {}
        def request(edits=()):
            key = tuple(sorted(edits))
            if key not in lookup:
                lookup[key] = len(req); req.append((gi, key, n))
            return lookup[key]
        base = request()
        common = dict(fold=fold, organoid_str=org, observed_n=n, timepoint=meta['timepoint'], dataset=str(meta['dataset']))
        states = {b: source_state(x, a, b) for a, b in FAMILIES}
        for i in range(n):
            census.append(common | dict(node=i, region=regions[i], crypt_id=int(labels[i]), curvature=float(curvature[i]),
                **{m: int(x[i, j] > .5) for j, m in enumerate(MARKERS)},
                lyso_state=states['Lysozyme'][i], sero_state=states['Serotonin'][i]))
        selected = set()
        for precursor, mature in FAMILIES:
            for state in ['precursor_only', 'double_positive', 'mature_only']:
                eligible = np.flatnonzero(states[mature] == state)
                chosen = rng.choice(eligible, min(cap, len(eligible)), replace=False)
                for src in sorted(chosen):
                    for marker in [precursor, mature]:
                        mi = MARKERS.index(marker)
                        if x[src, mi] > .5:
                            selected.add((int(src), mi, mature, state))
        for src, mi, family, state in sorted(selected):
            edit = request(((src, mi),))
            for hop, recipients in enumerate(neighborhoods[src], 1):
                for recipient in recipients:
                    near = sorted(set(neighborhoods[recipient][0] + neighborhoods[recipient][1]) - {src})
                    singles.append(common | dict(case_id=len(singles), source=src, recipient=recipient,
                        marker=MARKERS[mi], family=family, source_state=state, hop=hop,
                        source_region=regions[src], recipient_region=regions[recipient],
                        recipient_crypt_id=int(labels[recipient]), recipient_markers='|'.join(m for j,m in enumerate(MARKERS) if x[recipient,j] > .5),
                        recipient_lgr5=int(x[recipient,4] > .5), recipient_degree=len(adjacency[recipient]),
                        local_lgr5_fraction=float(x[near,4].mean()) if near else 0.,
                        backup_lyso=int(x[near,5].sum()), backup_sero=int(x[near,6].sum()),
                        source_lyso=int(x[src,5]), source_sero=int(x[src,6]),
                        source_curvature=float(curvature[src]), recipient_curvature=float(curvature[recipient]),
                        base_request=base, edit_request=edit))
        # Targeted, same-hop, distinct-source redundancy with an intact LGR5 recipient.
        for hop in [1, 2]:
            candidates = []
            for center in np.flatnonzero(x[:,4] > .5):
                ns = neighborhoods[center][hop-1]
                aa, bb = [s for s in ns if x[s,5] > .5], [s for s in ns if x[s,6] > .5]
                valid = [(a,b) for a in aa for b in bb if a != b]
                if valid:
                    candidates.append((int(center), valid, len(aa), len(bb)))
            indices = rng.choice(len(candidates), min(cap, len(candidates)), replace=False)
            for ci in sorted(indices):
                center, valid, na, nb = candidates[ci]
                for pi in sorted(rng.choice(len(valid), min(cap, len(valid)), replace=False)):
                    a,b = valid[pi]
                    pairs.append(common | dict(case_id=len(pairs), recipient=center, source_a=a, source_b=b, hop=hop,
                        recipient_region=regions[center], count_lyso=na, count_sero=nb,
                        cross_positive=int(x[a,6] > .5 or x[b,5] > .5),
                        base_request=base, a_request=request(((a,5),)), b_request=request(((b,6),)),
                        both_request=request(((a,5),(b,6)))))
    singles, pairs, census = pd.DataFrame(singles), pd.DataFrame(pairs), pd.DataFrame(census)
    for name, table in [('recipient_manifest', singles), ('pair_manifest', pairs), ('cell_census', census)]:
        table.to_csv(out / f'{name}.csv.gz', index=False)
    print(f'Prepared {len(singles)} recipient effects, {len(pairs)} pairs, {len(census)} cells', flush=True)
    return graphs, requests, singles, pairs


@torch.no_grad()
def predict(model, graphs, requests, sizes, device, batch_size=32):
    outputs = []
    for start in range(0, len(requests), batch_size):
        items=[]
        for (gi, edits, _), size in zip(requests[start:start+batch_size], sizes[start:start+batch_size]):
            g=copy.copy(graphs[gi]); g.x=g.x.clone()
            for node, marker in edits:
                assert g.x[node,marker] > .5
                g.x[node,marker]=0
            g.global_feat=g.x.new_tensor([[size]])
            items.append(g)
        batch=Batch.from_data_list(items).to(device)
        (mu,_),_=model(batch.x,batch.edge_index,batch)
        values=mu.reshape(-1).cpu().numpy().astype(float)
        offsets=batch.ptr.cpu().numpy()
        outputs.extend(values[offsets[i]:offsets[i+1]] for i in range(len(items)))
    return outputs


def run(run_dir, project_root, device='cuda', cap=3, sampling_data_dir=None):
    run_dir, project_root=Path(run_dir),Path(project_root)
    out=run_dir/'niche_hypotheses'; out.mkdir(exist_ok=True)
    settings=json.loads((run_dir/'settings.json').read_text())
    assert settings['NUM_LAYERS']==2, 'This analysis is defined for the saved depth-two model.'
    data_dir=project_root/'training_data'/settings['DATASET_NAME']
    assert json.loads(next(data_dir.glob('*_markers.json')).read_text())==MARKERS
    counts=json.loads((run_dir/'sweep_grid.json').read_text())
    config=dict(version=1, model='gin_film_size', cap_per_source_state_per_organoid=cap,
        sampling_seed=2301, counts=counts, seeds=settings['MODEL_SEEDS'], graph_context='full graph',
        intervention='one marker bit on one source; recipient unchanged', normalization='saved training-fold area regression / (4*pi)')
    if sampling_data_dir is not None:
        config['sampling_data_dir']=str(Path(sampling_data_dir).resolve())
        config['evaluation_data_dir']=str(data_dir.resolve())
    path=out/'settings.json'
    if path.exists(): assert json.loads(path.read_text()) == config
    path.write_text(json.dumps(config,indent=2))
    torch.set_num_threads(4)
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    sampling_dir=data_dir if sampling_data_dir is None else Path(sampling_data_dir)
    graphs,requests,singles,pairs=prepare(run_dir,sampling_dir,out,cap)
    if sampling_dir.resolve()!=data_dir.resolve():
        # Keep the fixed sampling cohorts and interventions, but restore the
        # evaluation encoding everywhere in the graph for a paired comparison.
        for group in graphs.values():
            for graph in group:
                with np.load(data_dir/f'{graph.organoid_str}.npz') as archive:
                    x=archive['x']
                assert x.shape==tuple(graph.x.shape)
                graph.x=torch.as_tensor(x,dtype=graph.x.dtype).clone()
    refs=pd.read_csv(run_dir/'geometric_normalization/references.csv').set_index('fold')
    for fold in sorted(graphs):
        s=singles[singles.fold==fold]; p=pairs[pairs.fold==fold]
        with (run_dir/f'checkpoints/fold_{fold}_preprocessing.pkl').open('rb') as f: pre=pickle.load(f)
        transform=pre['residual_transform']; r=refs.loc[fold]
        for seed in settings['MODEL_SEEDS']:
            prefix=out/f'fold_{fold}_seed_{seed}'
            done=Path(str(prefix)+'_done.json')
            if done.exists(): print(f'Skip {fold}/{seed}',flush=True); continue
            model=SizeFiLMGINCurvature(7,hidden_dim=settings['HIDDEN_DIM'],num_layers=2,global_dim=1,
                film_hidden_dim=settings['FILM_HIDDEN_DIM'],dropout=settings['DROPOUT'],norm=settings['NORM'],residual=settings['RESIDUAL'])
            model.load_state_dict(torch.load(run_dir/f'checkpoints/fold_{fold}_seed_{seed}_gin_film_size.pt',map_location='cpu',weights_only=True))
            model.to(device).eval()
            for n in counts+['observed']:
                tag=str(n)
                sp=Path(str(prefix)+f'_N{tag}_singles.csv.gz'); pp=Path(str(prefix)+f'_N{tag}_pairs.csv.gz')
                if sp.exists() and pp.exists(): continue
                ns=np.asarray([q[2] if n=='observed' else n for q in requests[fold]])
                sizes=(np.log(ns)-pre['size_center'])/pre['size_scale']
                z=predict(model,graphs[fold],requests[fold],sizes,device)
                def gather(frame,key):
                    return np.asarray([z[int(q)][int(c)] for q,c in zip(frame[key],frame.recipient)])
                zb,ze=gather(s,'base_request'),gather(s,'edit_request')
                raw=np.asarray(transform.inverse(ze))-np.asarray(transform.inverse(zb))
                n_s=s.observed_n.to_numpy() if n=='observed' else n
                scale=np.exp(r.alpha+r.beta*np.log(n_s))/(4*np.pi)
                pd.DataFrame(dict(case_id=s.case_id,delta_z=ze-zb,delta_raw=raw,delta_relative=raw*scale)).to_csv(sp,index=False)
                vv=[gather(p,k) for k in ['base_request','a_request','b_request','both_request']]
                rr=[np.asarray(transform.inverse(v)) for v in vv]
                n_p=p.observed_n.to_numpy() if n=='observed' else n
                scale_p=np.exp(r.alpha+r.beta*np.log(n_p))/(4*np.pi)
                table=pd.DataFrame(dict(case_id=p.case_id))
                for unit, values in [('z',vv),('raw',rr)]:
                    b,a,c,d=values
                    for name, value in [('lyso',a-b),('sero',c-b),('joint',d-b),('interaction',d-a-c+b)]:
                        table[f'{name}_{unit}']=value
                        if unit=='raw': table[f'{name}_relative']=value*scale_p
                table.to_csv(pp,index=False)
                print(f'Completed fold={fold} seed={seed} N={n}: {len(s)} recipients, {len(p)} pairs',flush=True)
            done.write_text(json.dumps(dict(fold=fold,seed=seed,n_single=len(s),n_pairs=len(p))))
    return out


def validate_saved(run_dir):
    """Check manifests, completion, and independently saved ego-graph predictions."""
    run_dir=Path(run_dir); out=run_dir/'niche_hypotheses'
    config=json.loads((out/'settings.json').read_text())
    s=pd.read_csv(out/'recipient_manifest.csv.gz'); p=pd.read_csv(out/'pair_manifest.csv.gz')
    assert (s.source!=s.recipient).all()
    assert ((p.source_a!=p.source_b)&(p.source_a!=p.recipient)&(p.source_b!=p.recipient)).all()
    assert s.case_id.is_unique and p.case_id.is_unique
    comparisons=0; differences={m:0. for m in ['delta_z','delta_raw','delta_relative']}
    for fold in sorted(s.fold.unique()):
        oldmeta=pd.read_json(run_dir/f'lgr5_film_diagnostics/fold_{fold}_single_manifest.json').rename(columns={
            'case_id':'old_case_id','orig_center':'recipient','orig_source_node':'source','source_marker_name':'marker'})
        match=s[s.fold==fold].merge(oldmeta[['old_case_id','organoid_str','recipient','source','marker','hop']],
            on=['organoid_str','recipient','source','marker','hop'],validate='one_to_one')
        for seed in config['seeds']:
            assert (out/f'fold_{fold}_seed_{seed}_done.json').exists()
            old=pd.read_csv(run_dir/f'lgr5_film_diagnostics/fold_{fold}_seed_{seed}_effects.csv.gz')
            old=old[old.route=='full'].rename(columns={'case_id':'old_case_id'})
            for n in config['counts']:
                new=pd.read_csv(out/f'fold_{fold}_seed_{seed}_N{n}_singles.csv.gz')
                assert len(new)==int((s.fold==fold).sum())
                assert np.isfinite(new[['delta_z','delta_raw','delta_relative']]).all().all()
                compared=match[['case_id','old_case_id']].merge(new,on='case_id').merge(
                    old[old.n==n][['old_case_id',*differences]],on='old_case_id',suffixes=('_new','_old'))
                comparisons+=len(compared)
                for metric in differences:
                    error=float((compared[metric+'_new']-compared[metric+'_old']).abs().max()) if len(compared) else 0.
                    differences[metric]=max(differences[metric],error)
                    np.testing.assert_allclose(compared[metric+'_new'],compared[metric+'_old'],atol=3e-6,rtol=3e-4)
                pair=pd.read_csv(out/f'fold_{fold}_seed_{seed}_N{n}_pairs.csv.gz')
                assert len(pair)==int((p.fold==fold).sum()) and np.isfinite(pair).all().all()
                for unit in ['z','raw','relative']:
                    np.testing.assert_allclose(pair[f'interaction_{unit}'],pair[f'joint_{unit}']-pair[f'lyso_{unit}']-pair[f'sero_{unit}'],atol=1e-12)
    result=dict(checkpoints=15,overlap_comparisons=comparisons,max_absolute_difference=differences,
        source_recipient_distinct=True,pair_sources_distinct=True,all_finite=True,factorial_identity=True,
        context_test='full graph agrees with independently saved two-hop recipient ego-graph effects')
    (out/'validation.json').write_text(json.dumps(result,indent=2))
    return result
