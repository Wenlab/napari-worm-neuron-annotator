"""Local proofreading-state performance acceptance benchmark.

Run from the repository root with ``pixi run python
scripts/benchmark_proofreading_state.py``.  Timings are deliberately reported
rather than asserted; correctness and operation counts are the CI gates.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

from napari_worm_neuron_annotator._proofread import ProofreadStore
from napari_worm_neuron_annotator._proofread_files import canonical_json_bytes
from napari_worm_neuron_annotator._roi import NeuronBoxDataset

RAW_SHAPE = (3315, 136, 8)
PATCH_COUNT = 34_000
REPETITIONS = 7


def _median_seconds(operation) -> float:
    operation()  # warm caches and allocator paths once
    samples = []
    for _ in range(REPETITIONS):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _build_store() -> ProofreadStore:
    raw = np.full(RAW_SHAPE, np.nan, dtype=np.float32)
    store = ProofreadStore(NeuronBoxDataset(raw))
    for index in range(PATCH_COUNT):
        store.set_observation_present(
            index // RAW_SHAPE[1],
            index % RAW_SHAPE[1],
            center_zyx=(3.0, 40.0 + index % 17, 50.0),
            size_zyx=(3.0, 7.0, 7.0),
        )
    return store


def _git_head_store_type():
    """Load the pre-change Store from Git HEAD when the worktree differs."""
    root = Path(__file__).resolve().parents[1]
    relative = "src/napari_worm_neuron_annotator/_proofread.py"
    try:
        source = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    if source == (root / relative).read_text(encoding="utf-8"):
        return None
    name = "napari_worm_neuron_annotator._proofread_git_head"
    module = types.ModuleType(name)
    module.__file__ = f"git:{relative}"
    module.__package__ = "napari_worm_neuron_annotator"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.ProofreadStore


def _current_load_cpu(dataset, payload) -> ProofreadStore:
    candidate = ProofreadStore(dataset)
    state = candidate._validate_payload(payload)  # noqa: SLF001
    candidate._install_working_state(  # noqa: SLF001
        state, advance_revision=False
    )
    candidate._set_saved_state(state)  # noqa: SLF001
    candidate._rebuild_dirty_cache()  # noqa: SLF001
    return candidate


def main() -> None:
    store = _build_store()
    plan = store._prepare_save()  # noqa: SLF001
    save_payload = plan.payload
    store._commit_saved_state(plan.committed_state)  # noqa: SLF001
    store.set_observation_present(
        0,
        0,
        center_zyx=(4.0, 40.0, 50.0),
        size_zyx=(3.0, 7.0, 7.0),
    )
    capture = store.capture_recovery_state()

    status_seconds = _median_seconds(
        lambda: (
            store.status.dirty,
            store.status.moved,
            store.status.resized,
            store.status.presence,
        )
    )
    capture_seconds = _median_seconds(store.capture_recovery_state)
    encode_seconds = _median_seconds(
        lambda: canonical_json_bytes(
            capture.payload(
                session_uuid="benchmark",
                revision=1,
                raw_sha256="0" * 64,
                utc_time="2026-09-13T00:00:00Z",
            )
        )
    )
    validate_seconds = _median_seconds(
        lambda: _current_load_cpu(store.dataset, save_payload)
    )

    print(f"shape={RAW_SHAPE}; patches={PATCH_COUNT}; median of {REPETITIONS}")
    print(f"cached status query: {status_seconds * 1000:.3f} ms")
    print(f"GUI-thread state capture: {capture_seconds * 1000:.3f} ms")
    print(f"worker payload + canonical JSON: {encode_seconds:.3f} s")
    print(f"validated typed-state load CPU: {validate_seconds:.3f} s")

    baseline_type = _git_head_store_type()
    if baseline_type is None:
        print("Git HEAD baseline unavailable or identical; comparison skipped")
        return

    def baseline_load_cpu():
        candidate = baseline_type(store.dataset)
        state = candidate._validate_payload(save_payload)
        candidate._restore_state(state)
        candidate._saved_snapshot = candidate._canonical_state()
        return candidate

    baseline = baseline_load_cpu()
    baseline.set_observation_present(
        0,
        0,
        center_zyx=(4.0, 40.0, 50.0),
        size_zyx=(3.0, 7.0, 7.0),
    )
    baseline_query_seconds = _median_seconds(
        lambda: (
            baseline.dirty,
            len(baseline.center_changed_observations),
            len(baseline.size_changed_observations),
            len(baseline.presence_changed_observations),
        )
    )
    baseline_capture_seconds = _median_seconds(
        lambda: baseline.recovery_payload(
            session_uuid="benchmark",
            revision=1,
            raw_sha256="0" * 64,
            utc_time="2026-09-13T00:00:00Z",
        )
    )
    baseline_load_seconds = _median_seconds(baseline_load_cpu)
    reduction = 100 * (1 - validate_seconds / baseline_load_seconds)
    print(f"Git HEAD status query: {baseline_query_seconds:.3f} s")
    print(
        "Git HEAD GUI recovery payload: "
        f"{baseline_capture_seconds:.3f} s"
    )
    print(f"Git HEAD load CPU: {baseline_load_seconds:.3f} s")
    print(f"load CPU reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()
