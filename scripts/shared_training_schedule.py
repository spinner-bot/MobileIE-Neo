"""Cooperative scheduling and read-only headroom guards for a bound training child."""
from collections import deque
import time

from bound_gpu_observer import BoundGPUObserver


class ResourcePressure(RuntimeError):
    pass


class SharedTrainingSchedule:
    def __init__(self, selected):
        self.selected = selected
        self.observer = BoundGPUObserver(selected['uuid'])
        self.samples = deque(maxlen=16)
        self.count = self.busy = 0
        self.work_seconds = self.pause_seconds = 0.
        self.minimum_free_mib = selected['total_mib']
        self.maximum_used_mib = self.maximum_utilization = 0
        self.last_utilization_decision = -float('inf')

    def check_memory(self):
        try:
            value = self.observer.read()
        except RuntimeError as error:
            raise ResourcePressure(f'Resource observation failed: {error}') from error
        self.samples.append(value)
        self.count += 1
        self.minimum_free_mib = min(self.minimum_free_mib, value['free_mib'])
        self.maximum_used_mib = max(self.maximum_used_mib, value['used_mib'])
        self.maximum_utilization = max(self.maximum_utilization, value['utilization_percent'])
        if value['free_mib'] < self.selected['incumbent_growth_reserve_mib'] + 1024:
            raise ResourcePressure('Incumbent memory growth reserve exhausted.')
        return value

    def yield_after_work(self, seconds):
        self.work_seconds += seconds
        self.check_memory()
        pause = max(.1, 3*seconds)
        time.sleep(pause)
        self.pause_seconds += pause
        value = self.check_memory()
        now = time.monotonic()
        # Distinct sustained decisions; NVML utilization is itself a sampled interval.
        if now - self.last_utilization_decision >= 2:
            self.busy = self.busy+1 if value['utilization_percent'] >= 90 else 0
            self.last_utilization_decision = now
        if self.busy >= 3:
            raise ResourcePressure('Three spaced post-yield utilization samples reached90%.')

    def summary(self):
        return dict(UUID=self.selected['uuid'], observations=self.count,
                    minimum_free_mib=self.minimum_free_mib, maximum_used_mib=self.maximum_used_mib,
                    maximum_observed_utilization_percent=self.maximum_utilization,
                    synchronized_work_seconds=self.work_seconds, cooperative_pause_seconds=self.pause_seconds,
                    current_consecutive_busy_decisions=self.busy, last16=list(self.samples),
                    scope='Lifetime resource minima/maxima and bounded recent observations; no phone or incumbent throughput claim.')
