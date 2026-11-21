"""One detached CPU worker; numerical completion is recorded by evaluate.py."""
import os
import subprocess
import time
from f1_io import BASE,ROOT,cache,identity,stamp,verify,write,read,digest


def main():
    verify()
    execution=read(BASE/'process/syntax-execution-freeze001.json')
    assert all(digest(ROOT/p)==h for p,h in execution['source_sha256'].items())
    assert not (BASE/'results/eval001').exists() and not (BASE/'process/launch002.json').exists()
    parent=cache().parent;parent.mkdir(parents=True,exist_ok=True)
    log=parent/'worker002.log'
    with log.open('x') as stream:
        worker=subprocess.Popen([str(ROOT/'scripts/project-python'),str(BASE/'evaluate_syntax001.py')],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            start_new_session=True)
    time.sleep(.02)
    write(BASE/'process/launch002.json',dict(at=stamp(),launcher=identity(),worker=identity(worker.pid),
        worker_session=os.getsid(worker.pid),log=str(log.relative_to(ROOT)),GPU_used=False,
        launcher_exit_does_not_prove_numerical_completion=True))
    print('F1 worker',worker.pid,'detached; log',str(log.relative_to(ROOT)))


if __name__=='__main__':main()
