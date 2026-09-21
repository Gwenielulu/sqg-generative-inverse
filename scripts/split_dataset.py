import argparse
import glob
import os
import shutil
from pathlib import Path
import numpy as np


def _split_dirs(base_dir):
    split_dirs = {
        "train": base_dir / "train",
        "valid": base_dir / "valid",
        "test": base_dir / "test",
    }
    for d in split_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        existing = list(d.glob("*.npy"))
        if existing:
            raise RuntimeError(f"{d} is not empty")
    return split_dirs


def _counts(n_total, train_ratio, valid_ratio, test_ratio):
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1")
    n_train = int(round(n_total * train_ratio))
    n_valid = int(round(n_total * valid_ratio))
    n_test = n_total - n_train - n_valid
    if n_train <= 0 or n_valid <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split counts for n_total={n_total}: train={n_train}, valid={n_valid}, test={n_test}"
        )
    return (n_train, n_valid, n_test)


def split_dataset(
    input_dir,
    output_dir,
    seed,
    train_ratio,
    valid_ratio,
    test_ratio,
    expected_total,
):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    split_dirs = _split_dirs(out_dir)
    npy_paths = sorted(glob.glob(os.path.join(in_dir, "*.npy")))
    if len(npy_paths) == 0:
        raise RuntimeError(f"No .npy files found in {input_dir}")
    n_total = len(npy_paths)
    if n_total != expected_total:
        raise ValueError(f"Expected {expected_total} trajectories, found {n_total}")
    rng = np.random.default_rng(seed)
    shuffled = [npy_paths[i] for i in rng.permutation(n_total)]
    n_train, n_valid, n_test = _counts(
        n_total=n_total,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )
    train_paths = shuffled[:n_train]
    valid_paths = shuffled[n_train : n_train + n_valid]
    test_paths = shuffled[n_train + n_valid :]
    assert len(test_paths) == n_test

    def _copy(paths, target_dir):
        for src in paths:
            src_path = Path(src)
            dst_path = target_dir / src_path.name
            shutil.copy2(src_path, dst_path)

    _copy(train_paths, split_dirs["train"])
    _copy(valid_paths, split_dirs["valid"])
    _copy(test_paths, split_dirs["test"])
    print("=" * 72)
    print("Split completed.")
    print(f"Total trajectories: {n_total}")
    print(f"Train: {len(train_paths)}")
    print(f"Valid: {len(valid_paths)}")
    print(f"Test : {len(test_paths)}")
    print(f"Output root: {out_dir}")
    print(f"  train -> {split_dirs['train']}")
    print(f"  valid -> {split_dirs['valid']}")
    print(f"  test  -> {split_dirs['test']}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Split .npy trajectories into train/valid/test"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing trajectory .npy files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output root directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Train split ratio"
    )
    parser.add_argument(
        "--valid_ratio", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--expected_total", type=int, default=50, help="Expected total trajectory count"
    )
    args = parser.parse_args()
    split_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        expected_total=args.expected_total,
    )


if __name__ == "__main__":
    main()
