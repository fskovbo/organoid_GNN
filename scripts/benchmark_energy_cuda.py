"""Benchmark CPU/CUDA parity and fitting using a saved mean-energy training run.

Writes a separate performance folder; existing checkpoints and analyses are read
only. No validation organoids are used in the matched fitting benchmark.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import argparse
import copy
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from src.artifacts.bundle import load_bundle,save_bundle
from src.artifacts.provenance import workflow_fingerprint
from src.artifacts.checkpoints import model_spec
from src.models.energy_ops import EnergyBatch
from src.training.coupled_fit import graph_samples
from src.training.energy_fit import EnergyObjective,TensorEnergyObjective,fit_energy,energy_mse


def timed(function,repeats=3):
    times=[];value=None
    for _ in range(repeats):
        torch.cuda.synchronize();start=time.perf_counter();value=function();torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
    return float(np.median(times)),value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--seed',type=int,help='Select a seed if the saved fold has multiple main models')
    parser.add_argument('--fit-organoids',type=int,default=256)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('A CUDA device is required for this benchmark')
    if args.repeats<1 or args.fit_organoids<2:raise ValueError('Need positive repeats and at least two fit organoids')
    torch.set_num_threads(4)
    output=args.output or args.run/'performance'/('cuda_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    settings=dict(run=str(args.run.resolve()),fold=args.fold,fit_organoids=args.fit_organoids,repeats=args.repeats,
        gpu=torch.cuda.get_device_name(),torch_version=str(torch.__version__),precision='float64',cpu_threads=4,solver_rtol=1e-11,
        code_sha256=workflow_fingerprint(ROOT/'experiments/training/mean_curvature_energy_training.ipynb'))
    (output/'settings.json').write_text(json.dumps(settings,indent=2))
    records=json.loads((args.run/'models.json').read_text())
    matches=[r for r in records if r['name']=='learned_accommodated' and r['fold']==args.fold and (args.seed is None or r['seed']==args.seed)]
    if len(matches)!=1:raise ValueError('Select one learned_accommodated checkpoint with --fold and, if needed, --seed')
    saved=load_bundle(args.run/matches[0]['bundle']);model=saved['model']
    settings['reference_checkpoint']=matches[0]['key'];(output/'settings.json').write_text(json.dumps(settings,indent=2))
    start=time.perf_counter();data=load_bundle(args.run/f'inputs/fold_{args.fold}_mean')
    samples=graph_samples(data['groups']['train'],2);preparation=time.perf_counter()-start
    del data
    metrics=dict(organoids=len(samples),nodes=sum(len(s['identity']) for s in samples),graph_preparation_seconds=preparation)
    rows=[]
    with threadpool_limits(limits=4):
        cpu=EnergyObjective(copy.deepcopy(model),samples)
        start=time.perf_counter();gpu=TensorEnergyObjective(copy.deepcopy(model),samples,device='cuda');torch.cuda.synchronize()
        metrics['gpu_setup_seconds']=time.perf_counter()-start
        p=cpu.pack();cpu(p);gpu(p)
        torch.cuda.reset_peak_memory_stats()
        ct,c=timed(lambda:cpu(p),args.repeats);gt,g=timed(lambda:gpu(p),args.repeats)
        metrics.update(objective_absolute_error=abs(c[0]-g[0]),gradient_max_absolute_error=float(np.max(abs(c[1]-g[1]))),coefficient_max_absolute_error=float(np.max(abs(cpu.model.weights.numpy()-gpu.model.weights.numpy()))))
        np.testing.assert_allclose(c[0],g[0],rtol=1e-10,atol=1e-11);np.testing.assert_allclose(c[1],g[1],rtol=1e-7,atol=1e-9)
        rows.append(dict(operation='objective_and_gradient',cpu_seconds=ct,cuda_seconds=gt,speedup=ct/gt));print(rows[-1],flush=True)
        ct,c=timed(cpu.sign_statistics,args.repeats);gt,g=timed(gpu.sign_statistics,args.repeats)
        for a,b in zip(c,g):np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-10)
        rows.append(dict(operation='sign_statistics',cpu_seconds=ct,cuda_seconds=gt,speedup=ct/gt));print(rows[-1],flush=True)
        ct,c=timed(cpu.select_signs,1);gt,g=timed(gpu.select_signs,1)
        np.testing.assert_array_equal(cpu.model.signs,gpu.model.signs)
        rows.append(dict(operation='complete_sign_search',cpu_seconds=ct,cuda_seconds=gt,speedup=ct/gt))
        ct,c=timed(lambda:model.predict_samples(samples),args.repeats)
        gt,g=timed(lambda:gpu.batch.predict(model).cpu().numpy(),args.repeats)
        metrics['prediction_max_absolute_error']=float(np.max(abs(np.concatenate(c)-g)))
        np.testing.assert_allclose(np.concatenate(c),g,rtol=1e-8,atol=1e-10)
        rows.append(dict(operation='prediction_cached_graphs',cpu_seconds=ct,cuda_seconds=gt,speedup=ct/gt))
        cold,_=timed(lambda:model.predict_samples(samples,device='cuda'),args.repeats)
        rows.append(dict(operation='prediction_including_gpu_preparation',cpu_seconds=ct,cuda_seconds=cold,speedup=ct/cold))
        metrics['peak_gpu_allocated_GB']=torch.cuda.max_memory_allocated()/1e9
        pd.DataFrame(rows).to_csv(output/'timings.csv',index=False)
        (output/'parity.json').write_text(json.dumps(metrics,indent=2))
        del cpu,gpu;torch.cuda.empty_cache()
        # Spread the fitting benchmark across the saved training fold's N range.
        order=np.argsort([s['N'] for s in samples],kind='stable')
        selected=order[np.linspace(0,len(order)-1,min(args.fit_organoids,len(order)),dtype=int)]
        fitting=[samples[i] for i in selected]
        (output/'fit_organoids.json').write_text(json.dumps([s['organoid_str'] for s in fitting],indent=2))
        model_settings=model_spec(model)['kwargs']
        penalty=saved['metrics']['best']['penalty'];fits={};fit_times={}
        for device in ['cpu','cuda']:
            start=time.perf_counter()
            fitted,info,trials=fit_energy(fitting,model_settings=model_settings,penalties=[penalty],starts=[(.15,3.,1)],device=device,blas_threads=4,callback=lambda x:print(f'{device}: {x}',flush=True))
            torch.cuda.synchronize();fit_times[device]=time.perf_counter()-start
            fits[device]=(fitted,info)
            save_bundle(output/f'fit_{device}',dict(model=fitted,metrics=info),splits=dict(train=[s['organoid_str'] for s in fitting]))
            trials.to_json(output/f'fit_{device}_trials.json',orient='records',indent=2)
            print(device,'fit seconds',fit_times[device],flush=True)
        a,ai=fits['cpu'];b,bi=fits['cuda'];ap=np.concatenate(a.predict_samples(fitting));bp=np.concatenate(b.predict_samples(fitting,device='cuda'))
        fit_parity=dict(cpu_objective=ai['optimizer']['objective'],cuda_objective=bi['optimizer']['objective'],
            cpu_inner_mse=ai['best']['inner_mse'],cuda_inner_mse=bi['best']['inner_mse'],
            sign_disagreements=int((a.signs!=b.signs).sum()),prediction_max_absolute_error=float(np.max(abs(ap-bp))),
            prediction_rmse_difference=float(np.sqrt(np.mean((ap-bp)**2))),inner_membership_equal=ai['inner_train']==bi['inner_train'] and ai['inner_validation']==bi['inner_validation'])
        if not fit_parity['inner_membership_equal']:raise AssertionError('Different inner memberships')
        np.testing.assert_allclose(ai['optimizer']['objective'],bi['optimizer']['objective'],rtol=1e-6,atol=1e-7)
        (output/'fit_parity.json').write_text(json.dumps(fit_parity,indent=2))
        rows.append(dict(operation=f'complete_fit_{len(fitting)}_organoids',cpu_seconds=fit_times['cpu'],cuda_seconds=fit_times['cuda'],speedup=fit_times['cpu']/fit_times['cuda']))
        pd.DataFrame(rows).to_csv(output/'timings.csv',index=False)
    lines=['# CUDA migration benchmark','',f"Hardware: {settings['gpu']}; PyTorch {settings['torch_version']}. Both backends use float64. CPU timings use four BLAS threads.",'',
        f"Steady-state checks use {metrics['organoids']} organoids / {metrics['nodes']:,} cells from saved training fold {args.fold}. Exact graph-shell preparation remains on CPU and took {preparation:.2f} s. GPU setup took {metrics['gpu_setup_seconds']:.2f} s.",'',
        '| Operation | CPU seconds | CUDA seconds | Speedup |','| --- | ---: | ---: | ---: |']
    lines += [f"| {r['operation']} | {r['cpu_seconds']:.4f} | {r['cuda_seconds']:.4f} | {r['speedup']:.2f}× |" for r in rows]
    lines += ['',f"Fixed-parameter agreement: objective {metrics['objective_absolute_error']:.3g}; maximum gradient error {metrics['gradient_max_absolute_error']:.3g}; maximum prediction error {metrics['prediction_max_absolute_error']:.3g}. Peak allocated GPU memory: {metrics['peak_gpu_allocated_GB']:.2f} GB.",'',
        f"The complete-fit benchmark uses {len(fitting)} organoids spread across training N, one penalty and one start, with identical inner selection and full refitting. It does not rerun the original five-fold experiment. Final objective: CPU {fit_parity['cpu_objective']:.10g}, CUDA {fit_parity['cuda_objective']:.10g}; sign disagreements {fit_parity['sign_disagreements']}; maximum fitted prediction difference {fit_parity['prediction_max_absolute_error']:.3g}.",'',
        'SciPy still performs the small nonlinear optimization and sequential sign choices on CPU. CUDA performs response construction, conditional reference means, propagated designs, linear profiling, analytic gradients and large sign-search statistics. Sparse CG verifies its true residual; failed solves raise an error. No float32 or mixed-precision approximation is used. Small inputs and fresh graph preparation can limit speedup. Existing scientific results were not overwritten.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n');(output/'complete.json').write_text(json.dumps(dict(parity_passed=True,fit_completed=True),indent=2))
    print('Benchmark report:',output/'REPORT.md',flush=True)


if __name__=='__main__':main()
