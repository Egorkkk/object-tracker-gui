"""GPU integration: terminate server mid-range, reopen and resume saved progress."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs' / ('recovery-' + str(time.time_ns()))
OUTPUT.mkdir(parents=True)
PROJECT = OUTPUT / 'project'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(route, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request('http://127.0.0.1:8766/api/' + route, data=data,
                                 headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=30) as response:
        return json.load(response)


def wait(predicate, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            state = request('state')
            if state.get('error'):
                raise RuntimeError(state['error'])
            if predicate(state):
                return state
        except (ConnectionError, urllib.error.URLError):
            pass
        time.sleep(.1)
    raise TimeoutError('Server state timeout')


def launch(log, reopen=False):
    args = [sys.executable, '-m', 'object_tracker.app.server', '--port', '8766']
    if reopen:
        args += ['--project', str(PROJECT)]
    process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    wait(lambda state: True)
    return process


process = None
try:
    with (OUTPUT / 'server.log').open('w') as log:
        process = launch(log)
        data = ROOT / 'testdata/phone_short'
        request('create', dict(directory=str(PROJECT), source=str(data/'frames'), mesh=str(data/'phone_mm.ply'),
                               camera=dict(width=1920,height=1080,fx=1867.,fy=1867.,cx=960.,cy=540.)))
        request('pose/import', dict(index=0, path=str(data/'initial_pose.npy')))
        request('track', dict(start=0,end=7))
        wait(lambda s: len((s['project']['job'] or {}).get('completed',[])) >= 1)
        process.terminate(); process.wait(timeout=30)
        saved = json.loads((PROJECT/'project.json').read_text())
        assert 0 < len(saved['job']['completed']) < 8
        preserved = {str(i): saved['poses'][str(i)] for i in saved['job']['completed']}
        process = launch(log, True)
        state = request('state')
        assert state['project']['job']['status'] == 'interrupted'
        request('resume', {})
        state = wait(lambda s: not s['busy'])
        assert state['project']['job']['status'] == 'completed'
        assert len(state['project']['job']['completed']) == 8
        for key, result in preserved.items():
            assert state['project']['poses'][key] == result
        metadata = state['project']['backend']['metadata']
        assert metadata['environment']['egl'] and metadata['environment']['libcuda']
        assert metadata['checkpoint']['sha256'] == 'f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086'
        report = dict(passed=True, preserved_frames=list(preserved), resumed_to=7, metadata=metadata)
        (OUTPUT/'report.json').write_text(json.dumps(report, indent=2))
        print('PASS', OUTPUT, flush=True)
finally:
    if process and process.poll() is None:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=30)
