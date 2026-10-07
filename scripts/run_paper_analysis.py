"""Execute one saved-model paper notebook with persistent outputs and an RSS guard."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    import nbformat
    from nbclient import NotebookClient
    path = ROOT / args.notebook
    document = nbformat.read(path, 4)
    def save(**kwargs):
        nbformat.write(document, path)
    def starting(cell_index, **kwargs):
        print(f'CELL {cell_index}: starting', flush=True)
    def finished(cell_index, **kwargs):
        save()
        print(f'CELL {cell_index}: completed', flush=True)
    client = NotebookClient(document, timeout=None, kernel_name='organoid-gnn',
        resources={'metadata': {'path': str(ROOT)}}, on_cell_start=starting,
        on_cell_executed=finished)
    try:
        client.execute()
    finally:
        save()


def supervise(args):
    import psutil
    path = ROOT / args.notebook
    output = ROOT/'paper_results/exclusive_lineages/paper_exclusive_20261006_153255/execution'
    output.mkdir(exist_ok=True)
    logfile = output/(path.stem+'.log')
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--notebook', args.notebook]
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    started = time.time(); peak = 0
    with logfile.open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                try:
                    family = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in family if p.is_running())
                    peak = max(peak, rss)
                    if rss > args.max_gib*2**30 or psutil.virtual_memory().available < 3*2**30:
                        raise RuntimeError(f'Analysis memory guard: {rss/2**30:.2f} GiB')
                except psutil.NoSuchProcess:
                    pass
                time.sleep(2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
            audit=dict(notebook=args.notebook,exit_code=process.returncode,elapsed_seconds=time.time()-started,peak_host_rss_gib=peak/2**30,log=str(logfile))
            (output/(path.stem+'.json')).write_text(json.dumps(audit,indent=2))
            print(json.dumps(audit),flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--notebook',required=True)
    parser.add_argument('--max-gib',type=float,default=8)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    worker(args) if args.worker else supervise(args)
