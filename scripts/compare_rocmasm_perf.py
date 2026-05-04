#!/usr/bin/env python3
"""Compare LLVM-backend and WaveASM rocmasm artifacts with rocprofv3.

The script consumes a run_gemm artifact directory containing:
  run_gemm_artifacts/simple_gemm_llvm.rocmasm
  run_gemm_artifacts/simple_gemm_waveasm.rocmasm
  run_gemm_artifacts/gemm_info.json
  run_gemm_artifacts/kernel_launch_info.json

It assembles both kernels, validates outputs, profiles each kernel with
rocprofv3, and writes a JSON summary with average runtime and TFLOPS.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import wave_runtime


DTYPES = {
    "f16": torch.float16,
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "i32": torch.int32,
}

DTYPE_SIZES = {
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "i32": 4,
}


BACKENDS = {
    "llvm": {
        "asm_name": "simple_gemm_llvm.rocmasm",
        "kernargs": "A,B,C",
    },
    "waveasm": {
        "asm_name": "simple_gemm_waveasm.rocmasm",
        "kernargs": "A,B,C,M,N,K",
    },
}


def log(message: str) -> None:
    print(f"[compare_rocmasm_perf] {message}", file=sys.stderr, flush=True)


def find_tool(candidates: list[str]) -> str:
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        path = Path(candidate)
        if path.exists():
            return str(path)
    raise RuntimeError(f"none of these tools were found: {candidates}")


def compile_asm(asm_path: Path, out_dir: Path, target: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clang = find_tool(["amdclang++", "clang++", "/opt/rocm/llvm/bin/clang++"])
    obj = out_dir / f"{asm_path.stem}.o"
    hsaco = out_dir / f"{asm_path.stem}.hsaco"
    subprocess.run(
        [
            clang,
            "-x",
            "assembler",
            "-target",
            "amdgcn-amd-amdhsa",
            "-mcode-object-version=5",
            f"-mcpu={target}",
            "-mwavefrontsize64",
            "-c",
            str(asm_path),
            "-o",
            str(obj),
        ],
        check=True,
    )
    subprocess.run(
        [
            clang,
            "-target",
            "amdgcn-amd-amdhsa",
            "-Xlinker",
            "--build-id=sha1",
            "-o",
            str(hsaco),
            str(obj),
        ],
        check=True,
    )
    return {"clang": clang, "object": str(obj), "hsaco": str(hsaco)}


def load_metadata(artifact_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    run_dir = artifact_dir / "run_gemm_artifacts"
    gemm_info = json.loads((run_dir / "gemm_info.json").read_text())
    launch_info = json.loads((run_dir / "kernel_launch_info.json").read_text())
    return run_dir, gemm_info, launch_info


def load_backend_launch_info(run_dir: Path, backend: str) -> dict[str, Any]:
    backend_launch_info = run_dir / f"{backend}_kernel_launch_info.json"
    if backend_launch_info.exists():
        return json.loads(backend_launch_info.read_text())
    return json.loads((run_dir / "kernel_launch_info.json").read_text())


def get_grid(launch_info: dict[str, Any], gemm_info: dict[str, Any]) -> list[int]:
    if launch_info.get("grid") is not None:
        return [int(x) for x in launch_info["grid"]]
    shape = gemm_info["shape"]
    block = gemm_info["block"]
    return [
        (int(shape["M"]) + int(block["BLOCK_M"]) - 1) // int(block["BLOCK_M"]),
        (int(shape["N"]) + int(block["BLOCK_N"]) - 1) // int(block["BLOCK_N"]),
        1,
    ]


def get_dtype_size(gemm_info: dict[str, Any], name: str) -> int:
    dtype = gemm_info["dtype"][name]
    if dtype not in DTYPE_SIZES:
        raise ValueError(f"unsupported dtype for {name}: {dtype}")
    return DTYPE_SIZES[dtype]


def make_inputs(
    gemm_info: dict[str, Any], *, initialized: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = gemm_info["shape"]
    dtype = gemm_info["dtype"]
    if initialized:
        torch.manual_seed(0)
        a = torch.randn(
            shape["M"], shape["K"], dtype=DTYPES[dtype["A"]], device="cuda"
        )
        b = torch.randn(
            shape["N"], shape["K"], dtype=DTYPES[dtype["B"]], device="cuda"
        )
        c = torch.zeros(
            shape["M"], shape["N"], dtype=DTYPES[dtype["C"]], device="cuda"
        )
    else:
        a = torch.empty(
            shape["M"], shape["K"], dtype=DTYPES[dtype["A"]], device="cuda"
        )
        b = torch.empty(
            shape["N"], shape["K"], dtype=DTYPES[dtype["B"]], device="cuda"
        )
        c = torch.empty(
            shape["M"], shape["N"], dtype=DTYPES[dtype["C"]], device="cuda"
        )
    return a, b, c


def make_launch_info(
    hsaco: Path,
    launch_info: dict[str, Any],
    gemm_info: dict[str, Any],
):
    wave_runtime.load_hip_functions()
    _, gpu_func = wave_runtime.load_binary(
        str(hsaco), launch_info.get("func_name", "wave_kernel")
    )
    blocks = launch_info["blocks"]
    cluster_dims = launch_info["cluster_dims"]
    grid = get_grid(launch_info, gemm_info)
    return wave_runtime.KernelLaunchInfo(
        torch.cuda.current_stream().cuda_stream,
        gpu_func,
        int(launch_info["shared_memory_bytes"]),
        int(grid[0]),
        int(grid[1]),
        int(grid[2]),
        int(blocks[0]),
        int(blocks[1]),
        int(blocks[2]),
        int(cluster_dims[0]),
        int(cluster_dims[1]),
        int(cluster_dims[2]),
    )


def launch_backend(
    backend: str,
    hsaco: Path,
    launch_info: dict[str, Any],
    gemm_info: dict[str, Any],
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> None:
    launch = make_launch_info(hsaco, launch_info, gemm_info)
    shape = gemm_info["shape"]
    if backend == "llvm":
        kernel_args = [a.data_ptr(), b.data_ptr(), c.data_ptr()]
    elif backend == "waveasm":
        kernel_args = [
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            int(shape["M"]),
            int(shape["N"]),
            int(shape["K"]),
        ]
    else:
        raise ValueError(f"unknown backend: {backend}")
    wave_runtime.launch(
        launch,
        wave_runtime.Int64Vector(kernel_args),
        wave_runtime.Int64Vector([]),
        [],
        wave_runtime.Int64Vector([]),
    )


def check_correctness(
    backend: str,
    hsaco: Path,
    launch_info: dict[str, Any],
    gemm_info: dict[str, Any],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    a, b, c = make_inputs(gemm_info, initialized=True)
    launch_backend(backend, hsaco, launch_info, gemm_info, a, b, c)
    torch.cuda.synchronize()
    expected = torch.matmul(a.to(torch.float32), b.t().to(torch.float32))
    compare = c.to(torch.float32)
    diff = (compare - expected).abs()
    return {
        "allclose": bool(torch.allclose(compare, expected, rtol=rtol, atol=atol)),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def worker_main(args: argparse.Namespace) -> None:
    log(f"worker start backend={args.backend}")
    artifact_dir = Path(args.artifact_dir).resolve()
    run_dir, gemm_info, _launch_info = load_metadata(artifact_dir)
    launch_info = load_backend_launch_info(run_dir, args.backend)
    asm_path = run_dir / BACKENDS[args.backend]["asm_name"]
    backend_out_dir = artifact_dir / args.output_subdir / args.backend
    hsaco = backend_out_dir / f"{asm_path.stem}.hsaco"
    if not hsaco.exists():
        log(f"worker compiling missing hsaco from {asm_path}")
        compile_asm(asm_path, backend_out_dir, gemm_info["hardware"]["target_arch"])

    log("worker allocating inputs")
    a, b, c = make_inputs(gemm_info)
    log(f"worker warmup count={args.warmup}")
    for _ in range(args.warmup):
        launch_backend(args.backend, hsaco, launch_info, gemm_info, a, b, c)
    torch.cuda.synchronize()

    log(f"worker iterations count={args.iterations}")
    for _ in range(args.iterations):
        launch_backend(args.backend, hsaco, launch_info, gemm_info, a, b, c)
    torch.cuda.synchronize()
    log("worker done")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def pick_column(row: dict[str, str], names: list[str]) -> str | None:
    for name in names:
        if name in row:
            return name
    lowered = {key.lower(): key for key in row}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def find_trace_csv(out_dir: Path, prefix: str) -> Path:
    exact = out_dir / f"{prefix}_kernel_trace.csv"
    if exact.exists():
        return exact
    matches = sorted(out_dir.glob("**/*kernel*_trace*.csv"))
    if not matches:
        raise RuntimeError(f"no rocprof kernel trace CSV under {out_dir}")
    return matches[0]


def average_last_kernel_ms(trace_csv: Path, iterations: int) -> tuple[str, float]:
    rows = csv_rows(trace_csv)
    if not rows:
        raise RuntimeError(f"empty rocprof trace: {trace_csv}")
    sample = rows[0]
    kind_col = pick_column(sample, ["Kind", "kind"])
    name_col = pick_column(sample, ["Kernel_Name", "KernelName", "Kernel Name", "Name"])
    start_col = pick_column(sample, ["Start_Timestamp", "StartNs", "Start (ns)"])
    end_col = pick_column(sample, ["End_Timestamp", "EndNs", "End (ns)"])
    if name_col is None or start_col is None or end_col is None:
        raise RuntimeError(f"unexpected rocprof trace columns: {list(sample)}")

    by_kernel: dict[str, list[float]] = {}
    for row in rows:
        if kind_col and row.get(kind_col, "").strip().upper() != "KERNEL_DISPATCH":
            continue
        name = row.get(name_col, "").strip()
        if not name:
            continue
        duration_ns = float(row[end_col]) - float(row[start_col])
        if duration_ns >= 0:
            by_kernel.setdefault(name, []).append(duration_ns)
    if not by_kernel:
        raise RuntimeError(f"no kernel dispatch rows in {trace_csv}")
    kernel_name, durations = max(by_kernel.items(), key=lambda item: len(item[1]))
    if len(durations) < iterations:
        raise RuntimeError(
            f"only {len(durations)} dispatches for {kernel_name}, expected {iterations}"
        )
    tail = durations[-iterations:]
    return kernel_name, (sum(tail) / len(tail)) / 1e6


def tflops(gemm_info: dict[str, Any], time_ms: float) -> float:
    shape = gemm_info["shape"]
    flops = 2.0 * float(shape["M"]) * float(shape["N"]) * float(shape["K"])
    return flops / (time_ms / 1000.0) / 1.0e12


def profile_pytorch_reference(
    gemm_info: dict[str, Any], warmup: int, iterations: int
) -> dict[str, Any]:
    a, b, _c = make_inputs(gemm_info, initialized=True)
    a_ref = a.to(torch.float32)
    b_ref_t = b.to(torch.float32).t()
    shape = gemm_info["shape"]
    out = torch.empty(
        shape["M"], shape["N"], dtype=torch.float32, device="cuda"
    )

    for _ in range(warmup):
        torch.mm(a_ref, b_ref_t, out=out)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        torch.mm(a_ref, b_ref_t, out=out)
    end.record()
    torch.cuda.synchronize()

    time_ms = start.elapsed_time(end) / iterations
    return {
        "method": "torch.cuda.Event",
        "operation": "torch.mm(A.float(), B.float().T, out=C.float32)",
        "note": (
            "This matches the correctness reference, not necessarily the exact "
            "rocmasm input/output dtype contract."
        ),
        "time_ms": time_ms,
        "tflops": tflops(gemm_info, time_ms),
    }

def build_hip_launcher(out_root: Path) -> Path:
    src = out_root / "rocmasm_profile_launcher.cpp"
    exe = out_root / "rocmasm_profile_launcher"
    source = r'''
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>

#define CHECK_HIP(expr)                                                        \
  do {                                                                         \
    hipError_t err = (expr);                                                   \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": "      \
                << hipGetErrorString(err) << std::endl;                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  if (argc != 18) {
    std::cerr << "usage: " << argv[0]
              << " <hsaco> <backend> <M> <N> <K> <gridX> <gridY> <gridZ>"
              << " <blockX> <blockY> <blockZ> <sharedMemBytes>"
              << " <aElemBytes> <bElemBytes> <cElemBytes>"
              << " <warmup> <iterations>" << std::endl;
    return 2;
  }

  std::string hsaco = argv[1];
  std::string backend = argv[2];
  long long m = std::atoll(argv[3]);
  long long n = std::atoll(argv[4]);
  long long k = std::atoll(argv[5]);
  unsigned grid_x = static_cast<unsigned>(std::strtoul(argv[6], nullptr, 10));
  unsigned grid_y = static_cast<unsigned>(std::strtoul(argv[7], nullptr, 10));
  unsigned grid_z = static_cast<unsigned>(std::strtoul(argv[8], nullptr, 10));
  unsigned block_x = static_cast<unsigned>(std::strtoul(argv[9], nullptr, 10));
  unsigned block_y = static_cast<unsigned>(std::strtoul(argv[10], nullptr, 10));
  unsigned block_z = static_cast<unsigned>(std::strtoul(argv[11], nullptr, 10));
  unsigned shared_mem_bytes =
      static_cast<unsigned>(std::strtoul(argv[12], nullptr, 10));
  size_t a_elem_bytes = static_cast<size_t>(std::strtoull(argv[13], nullptr, 10));
  size_t b_elem_bytes = static_cast<size_t>(std::strtoull(argv[14], nullptr, 10));
  size_t c_elem_bytes = static_cast<size_t>(std::strtoull(argv[15], nullptr, 10));
  int warmup = std::atoi(argv[16]);
  int iterations = std::atoi(argv[17]);

  void *a = nullptr;
  void *b = nullptr;
  void *c = nullptr;
  CHECK_HIP(hipMalloc(&a, static_cast<size_t>(m) * static_cast<size_t>(k) *
                              a_elem_bytes));
  CHECK_HIP(hipMalloc(&b, static_cast<size_t>(n) * static_cast<size_t>(k) *
                              b_elem_bytes));
  CHECK_HIP(hipMalloc(&c, static_cast<size_t>(m) * static_cast<size_t>(n) *
                              c_elem_bytes));

  hipModule_t module = nullptr;
  hipFunction_t func = nullptr;
  CHECK_HIP(hipModuleLoad(&module, hsaco.c_str()));
  CHECK_HIP(hipModuleGetFunction(&func, module, "wave_kernel"));

  long long mm = m;
  long long nn = n;
  long long kk = k;
  void *llvm_args[] = {&a, &b, &c};
  void *waveasm_args[] = {&a, &b, &c, &mm, &nn, &kk};
  void **kernel_args = nullptr;
  if (backend == "llvm") {
    kernel_args = llvm_args;
  } else if (backend == "waveasm") {
    kernel_args = waveasm_args;
  } else {
    std::cerr << "unknown backend: " << backend << std::endl;
    return 2;
  }

  auto launch_once = [&]() -> hipError_t {
    return hipModuleLaunchKernel(func, grid_x, grid_y, grid_z, block_x, block_y,
                                 block_z, shared_mem_bytes, nullptr,
                                 kernel_args, nullptr);
  };

  for (int i = 0; i < warmup; ++i) {
    CHECK_HIP(launch_once());
  }
  CHECK_HIP(hipDeviceSynchronize());

  for (int i = 0; i < iterations; ++i) {
    CHECK_HIP(launch_once());
  }
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipModuleUnload(module));
  CHECK_HIP(hipFree(a));
  CHECK_HIP(hipFree(b));
  CHECK_HIP(hipFree(c));
  return 0;
}
'''
    if not src.exists() or src.read_text() != source:
        src.write_text(source)
    hipcc = find_tool(["hipcc", "/opt/rocm/bin/hipcc", "/opt/rocm-7.2.2/bin/hipcc"])
    subprocess.run([hipcc, str(src), "-O2", "-o", str(exe)], check=True)
    return exe


def profile_backend(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    artifact_dir = Path(args.artifact_dir).resolve()
    run_dir, gemm_info, _launch_info = load_metadata(artifact_dir)
    launch_info = load_backend_launch_info(run_dir, backend)
    out_root = artifact_dir / args.output_subdir
    out_dir = out_root / "rocprof" / backend
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{backend}_rocmasm"
    rocprof = find_tool(["rocprofv3"])
    launcher = build_hip_launcher(out_root)
    hsaco = out_root / backend / f"{BACKENDS[backend]['asm_name'].replace('.rocmasm', '.hsaco')}"
    shape = gemm_info["shape"]
    grid = get_grid(launch_info, gemm_info)
    blocks = launch_info["blocks"]
    cluster_dims = [int(x) for x in launch_info["cluster_dims"]]
    if any(cluster_dims):
        raise RuntimeError(
            f"profile launcher does not support nonzero cluster dims: {cluster_dims}"
        )
    cmd = [
        rocprof,
        "--kernel-trace",
        "--stats",
        "TRUE",
        "--output-format",
        "csv",
        "--output-directory",
        str(out_dir),
        "--output-file",
        prefix,
        "--",
        str(launcher),
        str(hsaco),
        backend,
        str(shape["M"]),
        str(shape["N"]),
        str(shape["K"]),
        str(grid[0]),
        str(grid[1]),
        str(grid[2]),
        str(blocks[0]),
        str(blocks[1]),
        str(blocks[2]),
        str(int(launch_info["shared_memory_bytes"])),
        str(get_dtype_size(gemm_info, "A")),
        str(get_dtype_size(gemm_info, "B")),
        str(get_dtype_size(gemm_info, "C")),
        str(args.warmup),
        str(args.iterations),
    ]
    env = os.environ.copy()
    env.setdefault("ROCPROFILER_LOG_LEVEL", "error")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=args.profile_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "rocprofv3 worker timed out\n"
            f"cmd: {' '.join(cmd)}\n"
            f"timeout_s: {args.profile_timeout}\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            "rocprofv3 worker failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    trace_csv = find_trace_csv(out_dir, prefix)
    kernel_name, avg_ms = average_last_kernel_ms(trace_csv, args.iterations)
    return {
        "kernel_name": kernel_name,
        "time_ms": avg_ms,
        "tflops": tflops(load_metadata(artifact_dir)[1], avg_ms),
        "trace_csv": str(trace_csv),
        "rocprof_dir": str(out_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-subdir", default="compare_rocmasm_perf")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--profile-timeout", type=int, default=120)
    parser.add_argument("--backend", choices=sorted(BACKENDS), default=None)
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.backend is None:
            raise ValueError("--worker requires --backend")
        worker_main(args)
        return

    artifact_dir = Path(args.artifact_dir).resolve()
    run_dir, gemm_info, _launch_info = load_metadata(artifact_dir)
    target = gemm_info["hardware"]["target_arch"]
    out_root = artifact_dir / args.output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "shape": gemm_info["shape"],
        "dtype": gemm_info["dtype"],
        "target": target,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rocmasm_timing_method": "rocprofv3 kernel trace",
        "backends": {},
    }
    
    if not args.skip_pytorch:
        results["pytorch_reference"] = profile_pytorch_reference(
            gemm_info, args.warmup, args.iterations
        )

    backends = [args.backend] if args.backend else list(BACKENDS)
    for backend in backends:
        info = BACKENDS[backend]
        asm_path = run_dir / info["asm_name"]
        if not asm_path.exists():
            raise FileNotFoundError(f"missing {backend} asm: {asm_path}")
        launch_info = load_backend_launch_info(run_dir, backend)
        build = compile_asm(asm_path, out_root / backend, target)
        hsaco = Path(build["hsaco"])
        correctness = check_correctness(
            backend, hsaco, launch_info, gemm_info, args.rtol, args.atol
        )
        profile = profile_backend(args, backend)
        results["backends"][backend] = {
            "asm": str(asm_path),
            "kernargs": info["kernargs"],
            **build,
            "correctness": correctness,
            "profile": profile,
        }

    if "llvm" in results["backends"] and "waveasm" in results["backends"]:
        llvm_tflops = results["backends"]["llvm"]["profile"]["tflops"]
        waveasm_tflops = results["backends"]["waveasm"]["profile"]["tflops"]
        results["speedup_waveasm_over_llvm"] = (
            waveasm_tflops / llvm_tflops if llvm_tflops else None
        )
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
