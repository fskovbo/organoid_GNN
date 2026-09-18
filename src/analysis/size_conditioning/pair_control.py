"""Whole-source-vector factorial control on the exact exclusive-eligible pairs."""
from src.artifacts import pickle_compat as artifact_pickle
import json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data.io import load_organoid_npz,build_pyg_graph
from src.analysis.spatial.niche_inference import predict
from src.analysis.size_conditioning.cohort_inputs import make_model
from src.analysis.conditioning.response_statistics import cluster_summary


def run(comparison,root,device='cuda',draws=1000):
    comparison,root=Path(comparison),Path(root)
    reference=Path(json.loads((comparison/'comparison_settings.json').read_text())['reference_run'])
    settings=json.loads((reference/'settings.json').read_text());counts=json.loads((reference/'sweep_grid.json').read_text())
    pm=pd.read_csv(comparison/'exclusive/niche_hypotheses/pair_manifest.csv.gz')
    refs=pd.read_csv(reference/'geometric_normalization/references.csv').set_index('fold')
    out=comparison/'full_fate_pairs';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);allorg=[]
    for fold in range(5):
        p=pm[pm.fold==fold].copy();graphs=[];requests=[];rows=[]
        for org,g in p.groupby('organoid_str',sort=True):
            arr=load_organoid_npz(str(root/'training_data'/settings['DATASET_NAME']/f'{org}.npz'),strict=True,target_indices=[0])
            graph=build_pyg_graph(arr['x'],arr['edges'],arr['y']);graphs.append(graph);gi=len(graphs)-1;lookup={}
            def request(sources=()):
                edits=tuple(sorted((int(source),int(marker)) for source in sources for marker in np.flatnonzero(arr['x'][source]>.5)))
                if edits not in lookup:
                    lookup[edits]=len(requests);requests.append((gi,edits,len(graph.x)))
                return lookup[edits]
            for row in g.to_dict('records'):
                a,b=row['source_a'],row['source_b'];assert a!=b and row['recipient'] not in [a,b]
                row.update(base_request=request(),a_request=request([a]),b_request=request([b]),both_request=request([a,b]));rows.append(row)
        p=pd.DataFrame(rows)
        with (reference/f'checkpoints/fold_{fold}_preprocessing.pkl').open('rb') as f:pre=artifact_pickle.load(f)
        r=refs.loc[fold];transform=pre['residual_transform']
        for seed in settings['MODEL_SEEDS']:
            model=make_model(settings,list(range(7)))
            model.load_state_dict(torch.load(reference/f'checkpoints/fold_{fold}_seed_{seed}_gin_film_size.pt',map_location='cpu',weights_only=True))
            model.to(device).eval()
            for n in counts+['observed']:
                path=out/f'fold_{fold}_seed_{seed}_N{n}.csv.gz'
                if path.exists():table=pd.read_csv(path)
                else:
                    ns=np.asarray([q[2] if n=='observed' else n for q in requests])
                    sizes=(np.log(ns)-pre['size_center'])/pre['size_scale']
                    z=predict(model,graphs,requests,sizes,device)
                    vv=[np.asarray([z[int(q)][int(c)] for q,c in zip(p[key],p.recipient)]) for key in ['base_request','a_request','b_request','both_request']]
                    rr=[np.asarray(transform.inverse(v)) for v in vv]
                    n_p=p.observed_n.to_numpy() if n=='observed' else n
                    factor=np.exp(r.alpha+r.beta*np.log(n_p))/(4*np.pi)
                    table=pd.DataFrame(dict(case_id=p.case_id))
                    for unit,v in [('z',vv),('raw',rr)]:
                        b,a,c,d=v
                        for name,values in [('lyso',a-b),('sero',c-b),('joint',d-b),('interaction',d-a-c+b)]:
                            table[f'{name}_{unit}']=values
                            if unit=='raw':table[f'{name}_relative']=values*factor
                    table.to_csv(path,index=False)
                joined=table.merge(p,on='case_id',validate='one_to_one').assign(seed=seed,n=0 if n=='observed' else n)
                metrics=[c for c in table if c!='case_id']
                for selection,mask in [('all',np.ones(len(joined),dtype=bool)),('one_each_in_hop',(joined.count_lyso==1)&(joined.count_sero==1)&(joined.cross_positive==0))]:
                    g=joined[mask].groupby(['organoid_str','recipient','hop','seed','n'])[metrics].mean().reset_index()
                    g=g.groupby(['organoid_str','hop','seed','n'])[metrics].mean().reset_index().assign(selection=selection)
                    allorg.append(g)
            print(f'Full-fate pair control {fold}/{seed} complete',flush=True)
    frame=pd.concat(allorg,ignore_index=True);frame.to_csv(out/'pairs_organoid.csv.gz',index=False)
    keys=['selection','hop','n'];metrics=[c for c in frame if c.endswith(('_z','_raw','_relative'))]
    cluster_summary(frame,keys,metrics,draws=draws).to_csv(out/'pairs_summary.csv',index=False)
    ix=['selection','hop','organoid_str','seed']
    endpoint=(frame[frame.n==791].set_index(ix)[metrics]-frame[frame.n==165].set_index(ix)[metrics]).reset_index()
    cluster_summary(endpoint,['selection','hop'],metrics,draws=draws).to_csv(out/'pairs_endpoint.csv',index=False)
    endpoint.groupby(['selection','hop','seed'])[metrics].mean().to_csv(out/'pairs_endpoint_seeds.csv')
    (out/'complete.json').write_text(json.dumps(dict(checkpoints=15,shared_pair_cases=len(pm),whole_source_vectors_cleared=True)))
    return out
