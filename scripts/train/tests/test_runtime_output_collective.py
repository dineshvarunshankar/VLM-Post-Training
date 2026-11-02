"""Delayed ranks must share one fresh-output decision before any writes."""
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous, output, existing):
    from scripts.train.shared.runtime import validate_output_directory
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        if rank == 1:
            time.sleep(0.4)
        try:
            validate_output_directory(Path(output), None)
        except ValueError as exc:
            if not existing:
                raise
            assert "nonempty" in str(exc)
        else:
            assert not existing, "existing output must be rejected on every rank"
            if rank == 0:
                (Path(output) / "tracking").mkdir()
        dist.barrier()
    finally:
        dist.destroy_process_group()


class RuntimeOutputCollectiveTests(unittest.TestCase):
    def test_delayed_rank_accepts_same_fresh_run(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            output.mkdir()
            mp.spawn(_worker, args=(str(Path(root)/"rdzv"), str(output), False), nprocs=2)

    def test_existing_output_rejected_by_every_rank(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            output.mkdir()
            (output / "checkpoint").write_text("preserve")
            mp.spawn(_worker, args=(str(Path(root)/"rdzv"), str(output), True), nprocs=2)
            self.assertEqual((output/"checkpoint").read_text(), "preserve")
