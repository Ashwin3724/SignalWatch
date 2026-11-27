"""Start the local demo with one command; first run installs dependencies."""
from pathlib import Path
import hashlib
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
import venv

root=Path(__file__).resolve().parent
os.chdir(root)
try:
    port=int(os.getenv('PORT','8010'))
    if not 1<=port<=65535:raise ValueError()
except ValueError:
    print('PORT must be a number from 1 to 65535.',file=sys.stderr);sys.exit(1)
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as check:
    try:check.bind(('127.0.0.1',port))
    except OSError:
        print(f'Port {port} is already in use. Windows: set PORT=8011 then run python run.py again.',file=sys.stderr)
        sys.exit(1)
    # This preflight improves errors but can't eliminate races with another process.
venv_dir=root/'.venv'
python=venv_dir/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
if not python.exists():
    print('Creating a local Python environment…',flush=True)
    venv.EnvBuilder(with_pip=True).create(venv_dir)
requirements=root/'requirements.txt';stamp=venv_dir/'signalwatch-requirements.sha256'
digest=hashlib.sha256(requirements.read_bytes()).hexdigest()
if not stamp.exists() or stamp.read_text().strip()!=digest:
    print('Installing dependencies (internet is required the first time)…',flush=True)
    try:subprocess.run([str(python),'-m','pip','install','-r',str(requirements)],check=True)
    except subprocess.CalledProcessError:
        print('Dependency installation failed; check your connection and retry.',file=sys.stderr);sys.exit(1)
    stamp.write_text(digest)
url=f'http://127.0.0.1:{port}/'
print('Training the synthetic experiment locally. This may take a little while…',flush=True)
proc=subprocess.Popen([str(python),'-m','uvicorn','app.main:app','--host','127.0.0.1','--port',str(port)])
try:
    # Wait for startup/training before opening; never open a stale app on an occupied port.
    for _ in range(180):
        if proc.poll() is not None:sys.exit(proc.returncode or 1)
        try:
            with urllib.request.urlopen(url+'api/report',timeout=1) as response:
                if response.status==200:break
        except Exception:time.sleep(1)
    else:
        print('Server did not become ready in time; check terminal output.',file=sys.stderr)
        proc.terminate();proc.wait(timeout=10);sys.exit(1)
    print(f'SignalWatch ready: {url}  (Ctrl+C to stop)',flush=True)
    if '--no-browser' not in sys.argv:webbrowser.open(url)
    proc.wait()
except KeyboardInterrupt:
    if proc.poll() is None:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
