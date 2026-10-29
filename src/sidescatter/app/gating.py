"""Gating results, worked out in the background (see jobs.py).

Views call ensure(ws, sample_ids) before drawing. It returns True when every
sample is ready to plot; otherwise it starts one job per missing sample and
returns False, and the view shows its last picture faded until `ready` fires
for that sample. Editing gates again before results arrive cancels the stale
jobs (cancel_stale), so only results for the newest gates are ever shown.
"""

from PySide6 import QtCore

from .jobs import run_job


class GatingService(QtCore.QObject):
    ready = QtCore.Signal(object, str)   # workspace, sample id whose results just arrived

    def __init__(self):
        super().__init__()
        self._pending = {}   # job key -> (token, job)

    def ensure(self, ws, sample_ids):
        """True if all these samples can be plotted now; else start working them out."""
        ok = True
        for sid in sample_ids:
            if sid not in ws.samples or ws.is_ready(sid):
                continue
            ok = False
            self._request(ws, sid)
        return ok

    def cancel_stale(self, ws=None):
        """Cancel jobs whose gates have changed since they started."""
        for key, (token, job) in list(self._pending.items()):
            _, ws_id, sid = key
            if ws is not None and ws_id != id(ws):
                continue
            current = ws.gating_token(sid) if ws is not None else None
            if job.active and token != current:
                job.cancel()
            if not job.active:
                self._pending.pop(key, None)

    def prepare(self, ws, sample_ids, title, then, key=None):
        """Work out every listed sample's gating in one job (with progress and
        Cancel), then call then() on the UI thread. For exports and tables."""
        todo = [sid for sid in sample_ids if sid in ws.samples and not ws.is_ready(sid)]
        snapshots = [(sid, *ws.gating_snapshot(sid)) for sid in todo]

        def work(job):
            out = []
            for i, (sid, token, strategy, sample) in enumerate(snapshots):
                job.progress(i / max(len(snapshots), 1), f"Calculating gates: {sid}")
                out.append((sid, token, ws.compute_results(strategy, sample)))
                _warm(ws, sid)
            return out

        def done(out):
            for sid, token, res in out:
                if ws.store_results(sid, token, res):
                    self.ready.emit(ws, sid)
            if all(ws.is_ready(sid) for sid in sample_ids if sid in ws.samples):
                then()
            else:  # gates changed while this ran: go again with the new ones
                self.prepare(ws, sample_ids, title, then, key)

        return run_job(title, work, on_done=done, key=key)

    def _request(self, ws, sid):
        key = ("gating", id(ws), sid)
        token = ws.gating_token(sid)
        pending = self._pending.get(key)
        if pending and pending[0] == token and pending[1].active:
            return pending[1]
        token, strategy, sample = ws.gating_snapshot(sid)

        def work(job):
            job.check_cancelled()
            res = ws.compute_results(strategy, sample)
            job.check_cancelled()
            _warm(ws, sid)
            return res

        def done(res):
            if self._pending.get(key, (None, None))[1] is job:
                self._pending.pop(key)
            if ws.store_results(sid, token, res):
                self.ready.emit(ws, sid)

        job = run_job(f"Calculating gates: {sid}", work, on_done=done, key=key, quiet=True)
        self._pending[key] = (token, job)
        return job


def _warm(ws, sid):
    """Compensate the events now, so drawing doesn't have to."""
    try:
        ws.compensated_events(sid)
    except KeyError:  # the sample was removed meanwhile
        pass


_service = None


def service():
    global _service
    if _service is None:
        _service = GatingService()
    return _service
