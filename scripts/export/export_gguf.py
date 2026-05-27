"""
scripts/export/export_gguf.py
==============================
Export Lokneeti-3B to GGUF format for llama.cpp / Ollama.

Requires:
  - llama.cpp installed (clone and build locally)
  - Merged model (not LoRA adapter — run train_qlora.py --merge first)

Steps:
  1. Merge LoRA adapter into base model (if not done)
  2. Convert to GGUF using llama.cpp convert script
  3. Quantize to Q4_K_M (recommended for 3B models)
  4. Generate Ollama Modelfile

Usage:
  python scripts/export/export_gguf.py \\
      --merged-model outputs/merged_model \\
      --llama-cpp /path/to/llama.cpp \\
      --output outputs/gguf/lokneeti-3b-q4.gguf
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lokneeti.utils.logging import get_logger

log = get_logger(__name__)

OLLAMA_MODELFILE_TEMPLATE = """\
# Ollama Modelfile for Lokneeti-3B
# Usage: ollama create lokneeti-3b -f Modelfile

FROM {gguf_path}

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 2048

SYSTEM \"\"\"
You are Lokneeti, a constitutional governance reasoning system developed by Vikhram Labs.
Your purpose is to analyze Indian public policy, detect constitutional risks, and reason
about democratic governance using structured Constitutional Chain Compression methodology.
You do not engage in casual conversation. You produce precise, structured governance analysis.
\"\"\"
"""


def run_command(cmd: list[str], cwd: str = ".") -> bool:
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
    if result.returncode != 0:
        log.error(f"Command failed with exit code {result.returncode}")
        return False
    return True


def export_gguf(
    merged_model_dir: Path,
    llama_cpp_dir: Path,
    output_gguf: Path,
    quantization: str = "q4_k_m",
) -> bool:
    """
    Convert a merged HuggingFace model to GGUF and quantize.

    Args:
        merged_model_dir: Directory of the merged (non-LoRA) model.
        llama_cpp_dir:    Path to compiled llama.cpp repo.
        output_gguf:      Output path for final GGUF file.
        quantization:     GGUF quantization type (q4_k_m recommended).
    """
    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    fp16_gguf = output_gguf.parent / "lokneeti-3b-fp16.gguf"

    # Step 1: Convert to FP16 GGUF
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Fallback to older script name
        convert_script = llama_cpp_dir / "convert.py"

    if not convert_script.exists():
        log.error(
            f"llama.cpp convert script not found at {convert_script}.\n"
            "Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
        )
        return False

    ok = run_command([
        sys.executable, str(convert_script),
        str(merged_model_dir),
        "--outfile", str(fp16_gguf),
        "--outtype", "f16",
    ])
    if not ok:
        return False

    log.info(f"✅ FP16 GGUF created: {fp16_gguf}")

    # Step 2: Quantize
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_cpp_dir / "quantize"  # Older name

    if not quantize_bin.exists():
        log.error(
            "llama-quantize binary not found. Build llama.cpp first:\n"
            "  cd llama.cpp && mkdir build && cd build && cmake .. && cmake --build ."
        )
        return False

    ok = run_command([
        str(quantize_bin),
        str(fp16_gguf),
        str(output_gguf),
        quantization.upper(),
    ])
    if not ok:
        return False

    log.info(f"✅ Quantized GGUF ({quantization}): {output_gguf}")
    log.info(f"   File size: {output_gguf.stat().st_size / 1e9:.2f} GB")
    return True


def generate_modelfile(gguf_path: Path, output_dir: Path) -> Path:
    """Generate an Ollama Modelfile for the GGUF model."""
    modelfile_path = output_dir / "Modelfile"
    content = OLLAMA_MODELFILE_TEMPLATE.format(gguf_path=gguf_path.resolve())
    with open(modelfile_path, "w") as f:
        f.write(content)
    log.info(f"✅ Ollama Modelfile: {modelfile_path}")
    log.info("   To use: ollama create lokneeti-3b -f " + str(modelfile_path))
    return modelfile_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Lokneeti-3B to GGUF")
    ap.add_argument("--merged-model", type=str, required=True)
    ap.add_argument("--llama-cpp",    type=str, required=True)
    ap.add_argument("--output",       type=str, default="outputs/gguf/lokneeti-3b-q4km.gguf")
    ap.add_argument("--quant",        type=str, default="q4_k_m",
                    help="Quantization: q4_k_m | q5_k_m | q8_0 | f16")
    args = ap.parse_args()

    merged_dir  = Path(args.merged_model)
    llama_dir   = Path(args.llama_cpp)
    output_gguf = Path(args.output)

    if not merged_dir.exists():
        log.error(
            f"Merged model not found: {merged_dir}\n"
            "Run: python scripts/train/train_qlora.py --merge"
        )
        sys.exit(1)

    success = export_gguf(merged_dir, llama_dir, output_gguf, args.quant)
    if success:
        generate_modelfile(output_gguf, output_gguf.parent)
        log.info("🎉 GGUF export complete!")
    else:
        log.error("GGUF export failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
