import argparse
from pathlib import Path
import glob
import numpy as np
from netCDF4 import Dataset as NCDataset
from tqdm import tqdm


def _assert_shape(t, c, h, w, expected_t, expected_c, expected_h, expected_w, src_name):
    if expected_t is not None and t != expected_t:
        raise ValueError(f"{src_name}: expected t={expected_t}, got {t}")
    if expected_c is not None and c != expected_c:
        raise ValueError(f"{src_name}: expected c={expected_c}, got {c}")
    if expected_h is not None and h != expected_h:
        raise ValueError(f"{src_name}: expected h={expected_h}, got {h}")
    if expected_w is not None and w != expected_w:
        raise ValueError(f"{src_name}: expected w={expected_w}, got {w}")


def _convert_one_nc(
    nc_path,
    output_dir,
    var_name,
    time_chunk,
    expected_t,
    expected_c,
    expected_h,
    expected_w,
):
    produced = 0
    stem = nc_path.stem
    with NCDataset(nc_path, "r") as nc:
        nc.set_auto_mask(False)
        if var_name not in nc.variables:
            raise KeyError(f"{nc_path}: missing variable '{var_name}'")
        var = nc.variables[var_name]
        shape = tuple(var.shape)
        if len(shape) != 5:
            raise ValueError(f"{nc_path}: expected a 5D trajectory chunk, got {shape}")
        ntraj, t, c, h, w = shape
        for traj_idx in range(ntraj):
            out_path = output_dir / f"{stem}_traj{traj_idx:04d}.npy"
            _assert_shape(
                t, c, h, w, expected_t, expected_c, expected_h, expected_w, str(nc_path)
            )
            arr = np.lib.format.open_memmap(
                out_path, mode="w+", dtype=np.float32, shape=(t, c, h, w)
            )
            for t0 in range(0, t, time_chunk):
                t1 = min(t, t0 + time_chunk)
                chunk = var[traj_idx, t0:t1, :, :, :]
                arr[t0:t1] = np.asarray(chunk, dtype=np.float32)
            del arr
            produced += 1
    return produced


def main():
    parser = argparse.ArgumentParser(
        description="Convert SQG .nc chunks to per-trajectory .npy"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing .nc files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write .npy trajectories",
    )
    parser.add_argument(
        "--var_name", type=str, default="pv", help="NetCDF variable name"
    )
    parser.add_argument(
        "--time_chunk", type=int, default=20, help="Timesteps per streaming read"
    )
    parser.add_argument(
        "--expected_t", type=int, default=1000, help="Expected trajectory length"
    )
    parser.add_argument("--expected_c", type=int, default=2, help="Expected channels")
    parser.add_argument("--expected_h", type=int, default=256, help="Expected height")
    parser.add_argument("--expected_w", type=int, default=256, help="Expected width")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nc_paths = sorted(glob.glob(str(input_dir / "*.nc")))
    if len(nc_paths) == 0:
        raise RuntimeError(f"No NetCDF files found in {input_dir}")
    total = 0
    print(f"Found {len(nc_paths)} NC files")
    print(f"Writing trajectories to: {output_dir}")
    print(f"Streaming time chunk: {args.time_chunk}")
    for nc_path in tqdm(nc_paths, desc="Converting"):
        produced = _convert_one_nc(
            nc_path=Path(nc_path),
            output_dir=output_dir,
            var_name=args.var_name,
            time_chunk=args.time_chunk,
            expected_t=args.expected_t,
            expected_c=args.expected_c,
            expected_h=args.expected_h,
            expected_w=args.expected_w,
        )
        total += produced
    print("\nConversion complete.")
    print(f"NC files: {len(nc_paths)}")
    print(f"Trajectory npy files: {total}")
    print(
        f"Each trajectory shape target: ({args.expected_t}, {args.expected_c}, {args.expected_h}, {args.expected_w})"
    )


if __name__ == "__main__":
    main()
