import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
from sqgturb import SQG, irfft2


def settings(args):
    n = int(args.resolution)
    preset = {
        64: {"dt": 1200.0, "diff_efold": 86400.0},
        256: {"dt": 90.0, "diff_efold": 86400.0 / 16.0},
    }
    dt = preset[n]["dt"]
    diff_efold = preset[n]["diff_efold"]
    norder = 8
    dealias = True
    symmetric = True
    dek = 0.0
    nsq = 0.0001
    f = 0.0001
    g = 9.8
    theta0 = 300.0
    h = 10000.0
    r = dek * nsq / f
    u = 30.0
    lr = np.sqrt(nsq) * h / f
    domain_l = 20.0 * lr
    tdiab = 10.0 * 86400.0
    scalefact = f * theta0 / g
    outputinterval = 3.0 * 3600.0
    model_timesteps = int(outputinterval / dt)
    if model_timesteps < 1:
        raise ValueError("outputinterval/dt must be >= 1")
    spinup_steps = int(args.spinup_days * 86400.0 / outputinterval)
    cfg = dict(
        n=n,
        dt=dt,
        diff_efold=diff_efold,
        norder=norder,
        dealias=dealias,
        symmetric=symmetric,
        nsq=nsq,
        f=f,
        u=u,
        h=h,
        r=r,
        tdiab=tdiab,
        scalefact=scalefact,
        domain_l=domain_l,
        precision="single",
        outputinterval=outputinterval,
        spinup_days=args.spinup_days,
        spinup_steps=spinup_steps,
        nsteps=args.nsteps,
        ntraj=args.ntraj,
        sigma_train=args.sigma_train,
        base_seed=args.base_seed,
        model_timesteps=model_timesteps,
    )
    return cfg


def _add_fields(nc, cfg, ntraj_local):
    n = cfg["n"]
    nsteps = cfg["nsteps"]
    nc.set_auto_mask(False)
    nc.N = n
    nc.dt = cfg["dt"]
    nc.outputinterval = cfg["outputinterval"]
    nc.spinup_days = cfg["spinup_days"]
    nc.nsteps = nsteps
    nc.ntraj = ntraj_local
    nc.sigma_train = cfg["sigma_train"]
    nc.scalefact = cfg["scalefact"]
    nc.diff_efold = cfg["diff_efold"]
    nc.diff_order = cfg["norder"]
    nc.dealias = int(cfg["dealias"])
    nc.symmetric = int(cfg["symmetric"])
    nc.createDimension("traj", ntraj_local)
    nc.createDimension("t", nsteps)
    nc.createDimension("z", 2)
    nc.createDimension("y", n)
    nc.createDimension("x", n)
    pvvar = nc.createVariable("pv", np.float32, ("traj", "t", "z", "y", "x"), zlib=True)
    pvvar.units = "normalized"
    tvar = nc.createVariable("t_seconds", np.float32, ("traj", "t"), zlib=True)
    xvar = nc.createVariable("x", np.float32, ("x",))
    yvar = nc.createVariable("y", np.float32, ("y",))
    xvar.units = "meters"
    yvar.units = "meters"
    xvar[:] = np.arange(0, cfg["domain_l"], cfg["domain_l"] / n, dtype=np.float32)
    yvar[:] = np.arange(0, cfg["domain_l"], cfg["domain_l"] / n, dtype=np.float32)
    return (pvvar, tvar)


def generate_chunk(
    traj_start, traj_end, out_dir, threads_per_proc, cfg, show_inner_progress=True
):
    os.environ["OMP_NUM_THREADS"] = str(threads_per_proc)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_proc)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_proc)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_proc)
    os.makedirs(out_dir, exist_ok=True)
    nsteps = cfg["nsteps"]
    spinup_days = cfg["spinup_days"]
    n = cfg["n"]
    final_path = os.path.join(
        out_dir,
        f"sqg_N{n}_3hrly_traj{traj_start:04d}-{traj_end - 1:04d}_{nsteps}step_spin{spinup_days}d.nc",
    )
    tmp_path = final_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    ntraj_local = traj_end - traj_start
    nc = Dataset(tmp_path, mode="w", format="NETCDF4_CLASSIC")
    pvvar, tvar = _add_fields(nc, cfg=cfg, ntraj_local=ntraj_local)
    nexp = 20
    xx = np.arange(0, 2.0 * np.pi, 2.0 * np.pi / n, dtype=np.float32)
    yy = np.arange(0, 2.0 * np.pi, 2.0 * np.pi / n, dtype=np.float32)
    xx, yy = np.meshgrid(xx, yy)
    blob = (np.sin(xx / 2) ** (2 * nexp) * np.sin(yy) ** nexp).astype(np.float32)
    worker_tag = f"chunk {traj_start:04d}-{traj_end - 1:04d}"
    traj_iter = range(traj_start, traj_end)
    if show_inner_progress:
        traj_iter = tqdm(
            traj_iter,
            desc=f"{worker_tag} traj",
            leave=False,
            dynamic_ncols=True,
            ascii=True,
        )
    local_idx = 0
    for traj in traj_iter:
        rs = np.random.RandomState(cfg["base_seed"] + traj)
        pv0 = rs.normal(0, 100.0, size=(2, n, n)).astype(np.float32)
        pv0[1] = pv0[1] + 2000.0 * blob
        pv0[0] -= pv0[0].mean()
        pv0[1] -= pv0[1].mean()
        model = SQG(
            pv0,
            nsq=cfg["nsq"],
            f=cfg["f"],
            U=cfg["u"],
            H=cfg["h"],
            r=cfg["r"],
            tdiab=cfg["tdiab"],
            dt=cfg["dt"],
            diff_order=cfg["norder"],
            diff_efold=cfg["diff_efold"],
            dealias=cfg["dealias"],
            symmetric=cfg["symmetric"],
            threads=threads_per_proc,
            precision=cfg["precision"],
            tstart=0,
        )
        model.timesteps = cfg["model_timesteps"]
        spinup_iter = range(cfg["spinup_steps"])
        if show_inner_progress and cfg["spinup_steps"] > 0:
            spinup_iter = tqdm(
                spinup_iter,
                desc=f"{worker_tag} traj{traj:04d} spinup",
                leave=False,
                dynamic_ncols=True,
                ascii=True,
            )
        for _ in spinup_iter:
            model.advance()
        if show_inner_progress and cfg["spinup_steps"] > 0:
            spinup_iter.close()
        step_iter = range(cfg["nsteps"])
        if show_inner_progress:
            step_iter = tqdm(
                step_iter,
                desc=f"{worker_tag} traj{traj:04d} integrate",
                leave=False,
                dynamic_ncols=True,
                ascii=True,
            )
        for k in step_iter:
            model.advance()
            pv = irfft2(model.pvspec).astype(np.float32)
            pv_norm = pv / cfg["sigma_train"]
            pvvar[local_idx, k] = pv_norm
            tvar[local_idx, k] = np.float32(model.t)
        if show_inner_progress:
            step_iter.close()
        local_idx += 1
        nc.sync()
    nc.close()
    os.replace(tmp_path, final_path)
    return final_path


def generate_parallel(
    out_dir, chunk_size, n_workers, threads_per_proc, cfg, show_inner_progress=True
):
    ntraj = cfg["ntraj"]
    chunks = [(s, min(ntraj, s + chunk_size)) for s in range(0, ntraj, chunk_size)]
    print(
        f"traj={ntraj}, chunk={chunk_size}, chunks={len(chunks)}, workers={n_workers}, threads/proc={threads_per_proc}"
    )
    out_files = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [
            ex.submit(
                generate_chunk,
                s,
                e,
                out_dir,
                threads_per_proc,
                cfg,
                show_inner_progress,
            )
            for s, e in chunks
        ]
        for fu in tqdm(as_completed(futs), total=len(futs), desc="Chunks done"):
            out_files.append(fu.result())
    print("Wrote chunk files:")
    for p in sorted(out_files):
        print(" ", p)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SQG trajectories in NetCDF chunks"
    )
    parser.add_argument(
        "--resolution", type=int, default=256, help="Horizontal grid size"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./sqg_chunks",
        help="Output directory for .nc chunks",
    )
    parser.add_argument(
        "--ntraj", type=int, default=50, help="Total number of trajectories"
    )
    parser.add_argument(
        "--nsteps", type=int, default=1000, help="Timesteps per trajectory"
    )
    parser.add_argument(
        "--spinup_days", type=int, default=300, help="Spin-up days before saving"
    )
    parser.add_argument(
        "--sigma_train", type=float, default=2660.0, help="Normalization divisor for pv"
    )
    parser.add_argument("--base_seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument(
        "--chunk_size", type=int, default=10, help="Trajectories per chunk file"
    )
    parser.add_argument(
        "--n_workers", type=int, default=8, help="Parallel worker processes"
    )
    parser.add_argument(
        "--threads_per_proc", type=int, default=1, help="OMP threads per process"
    )
    parser.add_argument(
        "--no_inner_progress",
        action="store_true",
        help="Disable trajectory/spinup/integration progress bars inside workers",
    )
    args = parser.parse_args()
    cfg = settings(args)
    generate_parallel(
        out_dir=args.out_dir,
        chunk_size=args.chunk_size,
        n_workers=args.n_workers,
        threads_per_proc=args.threads_per_proc,
        cfg=cfg,
        show_inner_progress=not args.no_inner_progress,
    )


if __name__ == "__main__":
    main()
