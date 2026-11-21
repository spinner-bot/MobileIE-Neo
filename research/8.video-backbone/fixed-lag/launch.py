"""One detached CPU worker; numerical completion is recorded by evaluate.py."""
import os
import subprocess
import time
from f1_io import BASE,ROOT,cache,identity,stamp,verify,write


def main():
    verify()
    assert not (BASE/'results/eval001').exists() and not (BASE/'process/launch001.json').exists()
    parent=cache().parent;parent.mkdir(parents=True,exist_ok=True)
    log=parent/'worker001.log'
    with log.open('x') as stream:
        worker=subprocess.Popen([str(ROOT/'scripts/project-python'),str(BASE/'evaluate.py')],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            start_new_session=True)
    time.sleep(.02)
    write(BASE/'process/launch001.json',dict(at=stamp(),launcher=identity(),worker=identity(worker.pid),
        worker_session=os.getsid(worker.pid),log=str(log.relative_to(ROOT)),GPU_used=False,
        launcher_exit_does_not_prove_numerical_completion=True))
    print('F1 worker',worker.pid,'detached; log',str(log.relative_to(ROOT)))


if __name__=='__main__':main()
