#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wave_lang.kernel.lang.global_symbols import GLOBAL_ADDRESS_SPACE
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import compute_grid, get_default_arch


DTYPE_LABEL_TO_TORCH = {
    "f16": torch.float16,
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "i32": torch.int32,
}


def load_dsl_module(dsl_path: Path):
    spec = importlib.util.spec_from_file_location("gemm_dsl_module", dsl_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load DSL module from {dsl_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_folder_name(
    family: str,
    m: int,
    n: int,
    k: int,
    a_dtype: str,
    b_dtype: str,
    c_dtype: str,
) -> str:
    return (
        f"{family}_m{m}_n{n}_k{k}_"
        f"a{a_dtype}_b{b_dtype}_c{c_dtype}"
    )


def make_unique_output_dir(output_root: Path, base_name: str) -> Path:
    candidate = output_root / base_name
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = output_root / f"{base_name}_run{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def newest_file(directory: Path, pattern: str) -> Path | None:
    matches = list(directory.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def make_hyperparams(dsl_module, args):
    return {
        dsl_module.ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        dsl_module.ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        dsl_module.ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        dsl_module.BLOCK_M: args.block_m,
        dsl_module.BLOCK_N: args.block_n,
        dsl_module.BLOCK_K: args.block_k,
        dsl_module.M: args.m,
        dsl_module.N: args.n,
        dsl_module.K: args.k,
    }


def make_compile_options(args, hyperparams, target_arch: str, dump_dir: Path):
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        dump_intermediates=str(dump_dir),
        print_ir_after=[],
    )
    options.device = "hip"
    options.target = target_arch
    return options


def force_recompile_for_artifacts():
    from wave_lang.kernel.wave import cache as wave_cache

    wave_cache.WAVE_ALWAYS_COMPILE = 1


def make_launch_info_dict(options) -> dict:
    launch_info = options.kernel_launch_info
    grid = None
    if launch_info.grid is not None:
        grid = compute_grid((), launch_info.grid)
    return {
        "grid": grid,
        "grid_str": launch_info.grid_str,
        "blocks": list(launch_info.blocks),
        "shared_memory_bytes": launch_info.shared_memory_bytes,
        "cluster_dims": list(launch_info.cluster_dims),
        "func_name": launch_info.func_name,
    }


def write_launch_info(output_dir: Path, backend: str, options) -> dict:
    launch_info = make_launch_info_dict(options)
    write_text(
        output_dir / f"{backend}_kernel_launch_info.json",
        json.dumps(launch_info, indent=2, sort_keys=True),
    )
    return launch_info


def compile_llvm_backend(dsl_module, args, target_arch: str, output_dir: Path):
    force_recompile_for_artifacts()
    _, wave_kernel = dsl_module.build_simple_gemm()
    with TemporaryDirectory(prefix="wave-llvm-dump-") as dump_dir_name:
        dump_dir = Path(dump_dir_name)
        vmfb_path = output_dir / "simple_gemm.vmfb"
        options = make_compile_options(
            args, make_hyperparams(dsl_module, args), target_arch, dump_dir
        )
        options.backend = "llvm"
        options.create_vmfb_file = str(vmfb_path)

        compiled = wave_compile(options, wave_kernel)
        write_text(output_dir / "wave_module.mlir", compiled.asm)

        ll_file = newest_file(dump_dir, "*.ll")
        if ll_file:
            write_text(output_dir / "simple_gemm_llvm.ll", ll_file.read_text())

        llvm_asm_file = newest_file(dump_dir, "*.rocmasm")
        if llvm_asm_file:
            llvm_asm = llvm_asm_file.read_text()
            write_text(output_dir / "simple_gemm_llvm.rocmasm", llvm_asm)

    launch_info = write_launch_info(output_dir, "llvm", options)
    write_text(
        output_dir / "kernel_launch_info.json",
        json.dumps(launch_info, indent=2, sort_keys=True),
    )
    return compiled, options


def compile_waveasm_backend(dsl_module, args, target_arch: str, output_dir: Path):
    force_recompile_for_artifacts()
    _, wave_kernel = dsl_module.build_simple_gemm()
    with TemporaryDirectory(prefix="wave-waveasm-dump-") as dump_dir_name:
        dump_dir = Path(dump_dir_name)
        waveasm_target = args.waveasm_target or target_arch
        options = make_compile_options(
            args, make_hyperparams(dsl_module, args), waveasm_target, dump_dir
        )
        options.backend = "asm"
        options.compile_to_asm = True
        # validate_options currently requires wave_runtime for backend="asm".
        options.wave_runtime = True

        compiled = wave_compile(options, wave_kernel)
        waveasm_asm = compiled.asm

        waveasm_dump = newest_file(dump_dir, "*.rocmasm")
        if waveasm_dump:
            waveasm_asm = waveasm_dump.read_text()

        write_text(output_dir / "simple_gemm_waveasm.rocmasm", waveasm_asm)
        write_launch_info(output_dir, "waveasm", options)
    return compiled, options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/gpfs/home/yukalee/wave/artifacts")
    parser.add_argument("--family", default="gemm")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--a-dtype", choices=DTYPE_LABEL_TO_TORCH, default="f16")
    parser.add_argument("--b-dtype", choices=DTYPE_LABEL_TO_TORCH, default="f16")
    parser.add_argument("--c-dtype", choices=DTYPE_LABEL_TO_TORCH, default="f32")
    parser.add_argument(
        "--dsl-path",
        default="/gpfs/home/yukalee/wave/artifacts/simple_gemm_dsl.py",
    )
    parser.add_argument(
        "--strict-waveasm",
        action="store_true",
        help="Fail the script if the WaveASM backend does not compile.",
    )
    parser.add_argument(
        "--waveasm-target",
        choices=("gfx90a", "gfx942", "gfx950", "gfx1250"),
        default="gfx90a",
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    dsl_path = Path(args.dsl_path).resolve()
    dsl_module = load_dsl_module(dsl_path)
    constraints, _ = dsl_module.build_simple_gemm()

    family = args.family
    folder_name = make_folder_name(
        family,
        args.m,
        args.n,
        args.k,
        args.a_dtype,
        args.b_dtype,
        args.c_dtype,
    )

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    out_dir = make_unique_output_dir(output_root, folder_name)
    out_dir.mkdir(parents=True)
    script_output_dir = out_dir / "run_gemm_artifacts"
    script_output_dir.mkdir(parents=True, exist_ok=True)

    target_arch = get_default_arch()
    device_props = torch.cuda.get_device_properties(torch.device("cuda"))
    info = {
        "name": folder_name,
        "family": family,
        "shape": {"M": args.m, "N": args.n, "K": args.k},
        "block": {
            "BLOCK_M": args.block_m,
            "BLOCK_N": args.block_n,
            "BLOCK_K": args.block_k,
        },
        "dtype": {
            "A": args.a_dtype,
            "B": args.b_dtype,
            "C": args.c_dtype,
        },
        "hardware": {
            "device_name": torch.cuda.get_device_name(0),
            "gcn_arch": getattr(device_props, "gcnArchName", None),
            "target_arch": target_arch,
            "multi_processor_count": getattr(device_props, "multi_processor_count", None),
            "shared_memory_per_block": getattr(device_props, "shared_memory_per_block", None),
            "shared_memory_per_block_optin": getattr(
                device_props, "shared_memory_per_block_optin", None
            ),
            "warp_size": getattr(device_props, "warp_size", None),
            "total_memory": getattr(device_props, "total_memory", None),
        },
        "constraints": [repr(c) for c in constraints],
    }
    write_text(
        script_output_dir / "gemm_info.json",
        json.dumps(info, indent=2, sort_keys=True),
    )
    write_text(script_output_dir / "dsl_source.py", dsl_path.read_text())

    manifest = {
        "outputs": {
            "shared_mlir": "wave_module.mlir",
            "llvm_ir": "simple_gemm_llvm.ll",
            "llvm_backend_asm": "simple_gemm_llvm.rocmasm",
            "waveasm_backend_asm": "simple_gemm_waveasm.rocmasm",
            "waveasm_target": args.waveasm_target or target_arch,
            "vmfb": "simple_gemm.vmfb",
            "kernel_info": "gemm_info.json",
            "launch_info": "kernel_launch_info.json",
            "llvm_launch_info": "llvm_kernel_launch_info.json",
            "waveasm_launch_info": "waveasm_kernel_launch_info.json",
        },
        "backend_status": {},
    }

    llvm_compiled, _ = compile_llvm_backend(
        dsl_module, args, target_arch, script_output_dir
    )
    manifest["backend_status"]["llvm"] = "ok"

    try:
        compile_waveasm_backend(dsl_module, args, target_arch, script_output_dir)
        manifest["backend_status"]["waveasm"] = "ok"
    except Exception as exc:
        error_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        write_text(script_output_dir / "waveasm_backend_error.txt", error_text)
        manifest["backend_status"]["waveasm"] = f"error: {exc}"
        if args.strict_waveasm:
            write_text(
                script_output_dir / "artifact_manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True),
            )
            raise

    if args.run:
        torch.manual_seed(0)
        a = torch.randn(
            args.m,
            args.k,
            dtype=DTYPE_LABEL_TO_TORCH[args.a_dtype],
            device="cuda",
        )
        b = torch.randn(
            args.n,
            args.k,
            dtype=DTYPE_LABEL_TO_TORCH[args.b_dtype],
            device="cuda",
        )
        c = torch.zeros(
            args.m,
            args.n,
            dtype=DTYPE_LABEL_TO_TORCH[args.c_dtype],
            device="cuda",
        )
        llvm_compiled(a, b, c)
        expected = torch.matmul(a.to(torch.float32), b.t().to(torch.float32))
        compare = c.to(torch.float32)
        max_abs_diff = (compare - expected).abs().max().item()
        run_info = {
            "backend": "llvm",
            "allclose": bool(torch.allclose(compare, expected, rtol=1e-2, atol=1e-2)),
            "max_abs_diff": max_abs_diff,
        }
        write_text(
            script_output_dir / "run_result.json",
            json.dumps(run_info, indent=2, sort_keys=True),
        )

    write_text(
        script_output_dir / "artifact_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True),
    )
    print(out_dir)


if __name__ == "__main__":
    main()
