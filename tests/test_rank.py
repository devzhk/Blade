import os
import shutil
import torch
import tempfile
import sys
import pathlib

# Guarantee project root on sys.path for direct execution
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualize_rank import list_ordered_files, compute_incremental_ranks

def make_particle(batch_vectors):
    """batch_vectors: list of 1D tensors -> returns (B, C, H, W) as (B, D) for simplicity"""
    return torch.stack(batch_vectors, dim=0)  # (B, D)

def test_incremental_rank_growth():
    tmpdir = tempfile.mkdtemp(prefix='ranktest_')
    try:
        # We create 3 iterations with z then x. Dimension D=5.
        # Iter 0 z: two independent vectors e1, e2 -> rank 2
        torch.save({'iteration':0, 'particles': make_particle([
            torch.tensor([1.,0.,0.,0.,0.]),
            torch.tensor([0.,1.,0.,0.,0.]),
        ])}, os.path.join(tmpdir, 'blade_iter_0_z.pt'))
        # Iter 0 x: adds one linear combination (e1+e2) -> rank stays 2
        torch.save({'iteration':0, 'particles': make_particle([
            torch.tensor([1.,1.,0.,0.,0.]),
        ])}, os.path.join(tmpdir, 'blade_iter_0_x.pt'))
        # Iter 1 z: adds e3 -> rank 3
        torch.save({'iteration':1, 'particles': make_particle([
            torch.tensor([0.,0.,1.,0.,0.])
        ])}, os.path.join(tmpdir, 'blade_iter_1_z.pt'))
        # Iter 1 x: adds e4, e4+e3 -> rank 4
        torch.save({'iteration':1, 'particles': make_particle([
            torch.tensor([0.,0.,0.,1.,0.]),
            torch.tensor([0.,0.,1.,1.,0.]),
        ])}, os.path.join(tmpdir, 'blade_iter_1_x.pt'))
        # Iter 2 z: adds vector in existing span (e2 + e4) -> rank stays 4
        torch.save({'iteration':2, 'particles': make_particle([
            torch.tensor([0.,1.,0.,1.,0.]),
        ])}, os.path.join(tmpdir, 'blade_iter_2_z.pt'))
        # Iter 2 x: adds e5 -> rank 5
        torch.save({'iteration':2, 'particles': make_particle([
            torch.tensor([0.,0.,0.,0.,1.]),
        ])}, os.path.join(tmpdir, 'blade_iter_2_x.pt'))

        files = list_ordered_files(tmpdir)
        rows = compute_incremental_ranks(files, tmpdir, center=False, tol=1e-10)
        ranks = [r['cumulative_rank'] for r in rows]
        assert ranks == [2,2,3,4,4,5], f"Unexpected rank progression: {ranks}"
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    # Simple manual invocation
    test_incremental_rank_growth()
    print("Rank test passed.")
