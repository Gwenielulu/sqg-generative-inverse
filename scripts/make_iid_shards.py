import argparse
from pathlib import Path
from netCDF4 import Dataset
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--variable", default="pv")
    parser.add_argument("--samples_per_shard", default=10000, type=int)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*.nc"))
    if not paths:
        raise FileNotFoundError(f"No NetCDF files found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    pending_count = 0
    shard_index = 0

    def flush(count):
        nonlocal pending, pending_count, shard_index
        combined = np.concatenate(pending, axis=0)
        while combined.shape[0] >= count:
            output = args.output_dir / f"shard_{shard_index:05d}.npy"
            np.save(output, combined[:count].astype(np.float32, copy=False))
            combined = combined[count:]
            shard_index += 1
        pending = [combined] if combined.shape[0] else []
        pending_count = combined.shape[0]

    for path in paths:
        with Dataset(path, "r") as dataset:
            variable = np.asarray(dataset.variables[args.variable][:], dtype=np.float32)
        if variable.ndim != 5:
            raise ValueError(
                f"{path}: expected (trajectory,time,channel,y,x), got {variable.shape}"
            )
        flattened = variable.reshape(-1, *variable.shape[2:])
        pending.append(flattened)
        pending_count += flattened.shape[0]
        if pending_count >= args.samples_per_shard:
            flush(args.samples_per_shard)
    if pending_count:
        combined = np.concatenate(pending, axis=0)
        np.save(
            args.output_dir / f"shard_{shard_index:05d}.npy",
            combined.astype(np.float32, copy=False),
        )
    print(f"Wrote {shard_index + int(pending_count > 0)} shards to {args.output_dir}")


if __name__ == "__main__":
    main()
