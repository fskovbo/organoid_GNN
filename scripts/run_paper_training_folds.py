"""Run paper training sequentially, one bounded process per fold or model.

The worker executes notebook cells with only runtime/resume switches overridden.
No training configuration or model implementation is changed by this launcher.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def subset_jobs(notebook, folds):
    """Read the explicit grid without importing ML libraries or loading data."""
    document = json.loads(Path(notebook).read_text())
    tree = ast.parse(''.join(document['cells'][4]['source']))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'SETTINGS' for t in node.targets))
    grid_keys = {'marker_subsets', 'subset_depths', 'depths', 'hidden_dims', 'seeds'}
    settings = {kw.arg: ast.literal_eval(kw.value) for kw in assignment.value.keywords if kw.arg in grid_keys}
    return [(fold, f'gin_{subset}_intact_d{depth}_h{width}_f{fold}_s{seed}')
            for fold in folds for subset in settings['marker_subsets']
            for depth in settings.get('subset_depths', {}).get(subset, settings['depths'])
            for width in settings['hidden_dims'] for seed in settings['seeds']]


def worker(args):
    import queue
    assert Path(queue.__file__).name == 'queue.py' and 'python3' in queue.__file__, queue.__file__
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing unintended CPU training.')
    torch.set_num_threads(4)
    notebook = ROOT / args.notebook
    document = json.loads(notebook.read_text())
    namespace = {'__name__': '__main__'}
    os.chdir(ROOT)
    for index, cell in enumerate(document['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if index == 4:
            if args.model_key:
                if 'ACTIVE_MODELS = None' not in source:
                    raise ValueError('This notebook does not support single-model execution.')
                source = source.replace('ACTIVE_MODELS = None', f'ACTIVE_MODELS = {[args.model_key]!r}')
            source = source.replace('ACTIVE_FOLDS = None', f'ACTIVE_FOLDS = [{args.fold}]')
            source = source.replace('RESUME_RUN = None', f'RESUME_RUN = {args.run!r}')
            source = source.replace('SHOW_PROGRESS = True', 'SHOW_PROGRESS = False')
            source = source.replace('PRINT_EPOCHS = False', 'PRINT_EPOCHS = True')
        print(f'CELL {index}: starting', flush=True)
        exec(compile(source, f'{notebook}:cell{index}', 'exec'), namespace)
        print(f'CELL {index}: complete', flush=True)
    source_hash = hashlib.sha256(notebook.read_bytes()).hexdigest()
    snapshot = Path(args.run)/(f'training_source_{source_hash[:12]}.ipynb' if args.model_key
                               else f'training_fold_{args.fold}.ipynb')
    for cell in document['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    document['metadata']['execution_record'] = {'run': args.run, 'source_sha256': source_hash,
        'note': 'Executed by sequential workers; per-job completion and memory usage are in execution.json.'}
    snapshot.write_text(json.dumps(document, indent=1))


def supervise(args):
    import psutil
    run = Path(args.run)
    (run/'logs').mkdir(parents=True, exist_ok=True)
    jobs = subset_jobs(ROOT/args.notebook, args.folds) if args.per_model else [(f, None) for f in args.folds]
    audit_path = run/'execution.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else []
    print(f'Planned {len(jobs)} sequential jobs; memory limit {args.max_gib:g} GiB per worker.', flush=True)
    for index, (fold, key) in enumerate(jobs, 1):
        label = key or f'fold_{fold}'
        log = run/'logs'/f'{label}.log'
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--notebook', args.notebook,
                   '--run', str(run), '--fold', str(fold)]
        if key:
            command += ['--model-key', key]
        env = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        with log.open('a') as handle:
            print(f'START {index}/{len(jobs)}: {label}', flush=True)
            started = time.monotonic()
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            peak = 0
            try:
                while child.poll() is None:
                    try:
                        processes = [psutil.Process(child.pid), *psutil.Process(child.pid).children(recursive=True)]
                        rss = sum(p.memory_info().rss for p in processes if p.is_running())
                    except psutil.NoSuchProcess:
                        continue
                    peak = max(peak, rss)
                    if len(processes) > 8 or rss > args.max_gib * 2**30 or psutil.virtual_memory().available < 3*2**30:
                        raise RuntimeError(f'Worker safety limit: {len(processes)} processes, {rss/2**30:.2f} GiB RAM')
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError(f'{label} exited {child.returncode}; see {log}')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        audit.append(dict(job=label, fold=fold, peak_host_rss_bytes=peak,
                          elapsed_seconds=time.monotonic()-started, status='completed'))
        temporary = audit_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(audit, indent=2))
        temporary.replace(audit_path)
        print(f'COMPLETED {index}/{len(jobs)}: {label}; peak RAM {peak/2**30:.2f} GiB', flush=True)
    if not (run/'complete.json').exists():
        raise RuntimeError('Requested folds finished but the full model grid is incomplete.')
    print('COMPLETE', run, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--fold', type=int)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--per-model', action='store_true', help='One fresh process per lineage-subset model.')
    parser.add_argument('--model-key', help='Single model key passed to a worker.')
    parser.add_argument('--max-gib', type=float, default=8)
    args = parser.parse_args()
    (worker if args.worker else supervise)(args)


if __name__ == '__main__':
    main()
