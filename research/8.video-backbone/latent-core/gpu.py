"""L1 registered group 0--3 admission after CPU preparation, with own task lock."""
import fcntl
from training_io import ROOT, write


def acquire_gpu(bound, out):
    import torch
    from prebind_shared_gpu import actual_device
    from shared_gpu_admission import sustained_samples, assess, snapshot
    from shared_training_schedule import SharedTrainingSchedule, ResourcePressure
    assert not torch.cuda.is_initialized()
    samples = sustained_samples()
    selected = next(r for r in assess(samples)["candidates"] if r["uuid"] == bound)
    write(out / "admission.json", dict(samples=samples, selected=selected, CPU_preparation_completed=True))
    if not selected["allowed"] or selected["index"] not in (0, 1, 2, 3):
        raise ResourcePressure("L1 requires current headroom in registered physical group0..3")
    lock = (ROOT / f'.cache/gpu-{selected["index"]}.lock').open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immediate = snapshot()
        write(out / "immediate-pre-CUDA.json", immediate)
        current = next(r for r in immediate["GPUs"] if r["uuid"] == bound)
        if current["total_mib"] - current["used_mib"] < selected["required_free_mib"]:
            raise ResourcePressure("Headroom changed before CUDA initialization")
        torch.cuda.set_per_process_memory_fraction(2048 / selected["total_mib"], 0)
        marker = torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        actual = actual_device(bound)
        write(out / "actual-device.json", actual)
        assert actual["matched"]
        del marker
        return lock, SharedTrainingSchedule(selected), selected
    except BlockingIOError as error:
        lock.close()
        raise ResourcePressure("Another project task holds this card") from error
    except BaseException:
        lock.close()
        raise
