"""Notebook progress for model grids; observation only, never fitted state."""
from contextlib import contextmanager
from contextvars import ContextVar
from html import escape
from pathlib import Path
from time import perf_counter
import pandas as pd

_ACTIVE_TASK = ContextVar('training_progress_task', default=None)


def log_epochs(verbose=None):
    """Keep historical logging outside a panel; honor an explicit override."""
    active = _ACTIVE_TASK.get()
    return bool(verbose) if verbose is not None else (active.print_epochs if active else True)


def report_epoch(event, callback=None):
    """Publish already-computed metrics without touching optimization or RNG."""
    active = _ACTIVE_TASK.get()
    if active is not None:
        active.update_epoch(event)
    if callback is not None:
        callback(dict(event))


def _duration(seconds):
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours}h {minutes:02d}m {seconds:02d}s' if hours else f'{minutes}m {seconds:02d}s'


class TrainingProgress:
    """One HTML/widget panel with separate model, baseline and copy counters.

    Wrap a task around fitting AND checkpoint saving: it is counted as complete
    only when both succeed. A context variable passes epoch events through
    existing baseline wrappers without putting UI objects in saved transforms.
    """
    def __init__(self, total_models, *, total_copies=0, total_baselines=0,
                 show=True, print_epochs=False, log_path=None):
        self.totals = dict(model=int(total_models), copied=int(total_copies), baseline=int(total_baselines))
        if any(value < 0 for value in self.totals.values()):
            raise ValueError('Progress totals must be nonnegative.')
        self.completed = dict.fromkeys(self.totals, 0)
        self.rows = []
        self.current = None
        self.event = {}
        self.print_epochs = print_epochs
        self.log_path = Path(log_path) if log_path is not None else None
        self.started = perf_counter()
        self.state = 'Preparing'
        self.message = ''
        self.widgets = None
        if show:
            import ipywidgets as widgets
            from IPython import get_ipython
            from IPython.display import display
            self.widgets = dict(
                overall=widgets.IntProgress(min=0, max=max(1, total_models), description='Models'),
                epoch=widgets.IntProgress(min=0, max=1, description='Epoch'),
                status=widgets.HTML(), detail=widgets.HTML(),
                table=widgets.HTML(layout=widgets.Layout(max_height='260px', overflow='auto')))
            self.panel = widgets.VBox(list(self.widgets.values()))
            if get_ipython() is not None:
                display(self.panel)
        self._refresh()

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        if typ is not None:
            self.state = 'Interrupted' if issubclass(typ, KeyboardInterrupt) else 'Failed'
            self.message = str(value)
        elif all(self.completed[k] == self.totals[k] for k in self.totals):
            self.state = 'Complete'
            self.message = ''
        else:
            self.state = 'Stopped before all tasks completed'
        self._refresh()
        return False

    def stage(self, message):
        self.message = message
        self._refresh()

    @contextmanager
    def task(self, *, kind='model', **metadata):
        if self.current is not None:
            raise RuntimeError('Progress tasks must not overlap.')
        if kind not in self.totals:
            raise ValueError(f'Unknown progress task kind: {kind}')
        # A legacy masking reference may need a comparison baseline fitted.
        if kind == 'baseline' and self.completed[kind] == self.totals[kind]:
            self.totals[kind] += 1
        if self.completed[kind] >= self.totals[kind]:
            raise ValueError(f'More {kind} tasks than declared.')
        self.current = dict(kind=kind, **metadata)
        self.event = {}
        self.task_started = perf_counter()
        self.state = 'Copying checkpoint' if kind == 'copied' else 'Training'
        self.message = ''
        token = _ACTIVE_TASK.set(self)
        status = 'completed'
        self._refresh()
        try:
            yield self
        except BaseException as exc:
            status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            self.state = status.capitalize()
            self.message = str(exc)
            raise
        finally:
            _ACTIVE_TASK.reset(token)
            row = dict(self.current, status=status, epochs=self.event.get('epoch', 0),
                       metric=self.event.get('metric'), best_value=self.event.get('best_value'),
                       best_epoch=self.event.get('best_epoch'), elapsed_seconds=perf_counter()-self.task_started)
            self.rows.append(row)
            if status == 'completed':
                self.completed[kind] += 1
                self.state = 'Between tasks'
            self.current = None
            self.event = {}
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(self.rows).to_csv(self.log_path, index=False)
            self._refresh(table=True)

    def update_epoch(self, event):
        if self.current is None:
            raise RuntimeError('An epoch update requires an active task.')
        self.event = dict(event)
        self._refresh()

    def _refresh(self, *, table=False):
        done, total = self.completed['model'], self.totals['model']
        status = (f'<b>{escape(self.state)}</b> — {done} / {total} models trained; '
                  f'{total-done} remaining (including any active model). '
                  f'Baselines: {self.completed["baseline"]}/{self.totals["baseline"]}. '
                  f'Copied/reused checkpoints: {self.completed["copied"]}/{self.totals["copied"]}. '
                  f'Total elapsed: {_duration(perf_counter()-self.started)}.')
        detail = escape(self.message)
        if self.current:
            description = ' | '.join(f'{escape(str(k))}: {escape(str(v))}' for k,v in self.current.items())
            detail = f'<b>{description}</b><br>Current task elapsed: {_duration(perf_counter()-self.task_started)}<br>{detail}'
            if self.event:
                e = self.event
                detail += (f'<br>Epoch {e["epoch"]}/{e["max_epochs"]} (maximum; early stopping may finish sooner). '
                           f'{escape(e["metric"])}: {e["value"]:.6g}. '
                           f'Best: {e["best_value"]:.6g} at epoch {e["best_epoch"]}. '
                           f'Patience used: {e["bad_epochs"]}/{e["patience"]}.')
        self.status_html, self.detail_html = status, detail
        if self.widgets:
            w = self.widgets
            w['status'].value, w['detail'].value = status, detail
            w['overall'].value = done
            w['overall'].bar_style = 'danger' if self.state in ('Failed','Interrupted') else ('success' if self.state=='Complete' else '')
            w['epoch'].value = 0  # Reset before changing max, including between tasks.
            w['epoch'].max = max(1, self.event.get('max_epochs', 1))
            w['epoch'].value = self.event.get('epoch', 0)
            if table:
                w['table'].value = pd.DataFrame(self.rows).to_html(index=False, escape=True, float_format=lambda x:f'{x:.5g}')

    def close(self):
        if self.widgets:
            self.panel.close()
            for widget in self.widgets.values():
                widget.close()
