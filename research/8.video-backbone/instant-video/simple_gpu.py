"""Fresh admission plus cooperative monitoring, without changing other GPU tasks."""
import fcntl
import os
import sys
import time

from simple_io import ROOT, write
sys.path.insert(0, str(ROOT / 'scripts'))
from bound_gpu_observer import BoundGPUObserver
from prebind_shared_gpu import actual_device
from shared_gpu_admission import assess, snapshot, sustained_samples
from shared_training_schedule import ResourcePressure


class Schedule:
    def __init__(self, selected, qualification):
        self.selected = selected
        self.observer = BoundGPUObserver(selected['uuid'])
        self.qualification = qualification
        self.shared = True
        self.last_scan = self.last_busy = -float('inf')
        self.busy = self.count = 0
        self.work = self.pause = 0.
        self.minimum_free = selected['total_mib']
        self.maximum_used = self.maximum_utilization = 0
        self.last_processes = []

    def check_memory(self):
        try:
            value = self.observer.read()
        except RuntimeError as error:
            raise ResourcePressure(str(error)) from error
        self.count += 1
        self.minimum_free = min(self.minimum_free, value['free_mib'])
        self.maximum_used = max(self.maximum_used, value['used_mib'])
        self.maximum_utilization = max(self.maximum_utilization, value['utilization_percent'])
        if value['free_mib'] < self.selected['incumbent_growth_reserve_mib'] + 1024:
            raise ResourcePressure('Incumbent memory growth reserve exhausted')
        now = time.monotonic()
        if now - self.last_scan >= 5:
            row = next(r for r in snapshot()['GPUs'] if r['uuid'] == self.selected['uuid'])
            self.last_processes = row['processes']
            self.shared = any(int(p.split(',')[1].strip()) != os.getpid() for p in self.last_processes)
            self.last_scan = now
        return value

    def after_work(self, seconds):
        self.work += seconds
        self.check_memory()
        if self.shared or self.qualification:
            pause = max(.1, 3 * seconds)
            time.sleep(pause)
            self.pause += pause
            value = self.check_memory()
            now = time.monotonic()
            if now - self.last_busy >= 2:
                self.busy = self.busy + 1 if value['utilization_percent'] >= 90 else 0
                self.last_busy = now
            if self.busy >= 3:
                raise ResourcePressure('Three spaced shared post-yield busy samples reached90%')
        else:
            self.busy = 0

    def summary(self):
        return dict(uuid=self.selected['uuid'], observations=self.count,
                    minimum_free_mib=self.minimum_free, maximum_used_mib=self.maximum_used,
                    maximum_utilization_percent=self.maximum_utilization,
                    synchronized_student_seconds=self.work, cooperative_pause_seconds=self.pause,
                    other_compute_processes_seen_in_latest_scan=self.shared,
                    latest_process_rows=self.last_processes,
                    scope='Aggregate observations and retained latest scan, not incumbent throughput or peak own-process memory')


def acquire(bound, out, qualification):
    import torch
    assert not torch.cuda.is_initialized()
    samples = sustained_samples()
    selected = next(r for r in assess(samples)['candidates'] if r['uuid'] == bound)
    write(out / 'admission.json', dict(samples=samples, selected=selected, CPU_preparation_completed=True))
    if not selected['allowed'] or selected['index'] not in (1, 2):
        raise ResourcePressure('I1-S requires fresh headroom on registered physical1/2')
    lock = (ROOT / f'.cache/gpu-{selected["index"]}.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immediate = snapshot()
        write(out / 'immediate-pre-CUDA.json', immediate)
        current = next(r for r in immediate['GPUs'] if r['uuid'] == bound)
        if current['total_mib'] - current['used_mib'] < selected['required_free_mib']:
            raise ResourcePressure('Headroom changed immediately before CUDA initialization')
        torch.cuda.set_per_process_memory_fraction(2048 / selected['total_mib'], 0)
        marker = torch.empty(1, device='cuda')
        torch.cuda.synchronize()
        actual = actual_device(bound)
        write(out / 'actual-device.json', actual)
        assert actual['matched']
        del marker
        return lock, Schedule(selected, qualification)
    except BlockingIOError as error:
        lock.close()
        raise ResourcePressure('Another project task holds this card') from error
    except BaseException:
        lock.close()
        raise
