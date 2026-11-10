"""Check whether the local TwiBot-20 HGT checkpoint can consume the local builder output."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import marshal
from pathlib import Path
import re
import zipfile

import torch


def digest_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return digest_stream(stream)


def stack_width(path: Path, assignment_name: str, function_name: str | None) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    search = tree.body
    if function_name is not None:
        search = next(node.body for node in tree.body if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name)
    for node in ast.walk(ast.Module(body=search, type_ignores=[])):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                                                 and target.id == assignment_name
                                                 for target in node.targets):
            call = node.value
            if (not isinstance(call, ast.Call) or not call.args
                    or not isinstance(call.args[0], (ast.List, ast.Tuple))):
                raise ValueError(f"Cannot statically count {assignment_name} fields")
            return len(call.args[0].elts)
    raise ValueError(f"Cannot find {assignment_name} in {path}")


def return_stack_width(path: Path, function_name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == function_name)
    for node in ast.walk(function):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            call = node.value
            if call.args and isinstance(call.args[0], (ast.List, ast.Tuple)):
                return len(call.args[0].elts)
    raise ValueError(f"Cannot statically count return fields of {function_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-zip", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--builder-cache", type=Path, required=True)
    parser.add_argument("--manuscript-archives", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = json.loads((args.qwen_dir / "config.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(args.official_zip) as archive:
        with archive.open("Twibot-20/train.json") as stream:
            official_train_hash = digest_stream(stream)
    model_weights_hash = digest_file(args.qwen_dir / "model.safetensors")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_width = int(checkpoint["in_channels"]["user"])
    builder_numeric_width = stack_width(args.builder, "raw_numeric", "main")
    evaluator_numeric_width = return_stack_width(args.evaluator, "numeric")
    text_width = int(config["hidden_size"])
    builder_width = 2 * text_width + builder_numeric_width
    evaluator_width = 2 * text_width + evaluator_numeric_width
    build_log = args.build_log.read_text(encoding="utf-8", errors="replace")
    train_log = args.train_log.read_text(encoding="utf-8", errors="replace")
    build_log_match = re.search(r"final user feature dim: (\d+) \(desc=(\d+) \+ tweet=(\d+) \+ numeric=(\d+)\)", build_log)
    train_log_match = re.search(r"x=\[8278, (\d+)\]", train_log)
    if not build_log_match or not train_log_match:
        raise ValueError("Saved build/train logs lack the expected feature-width records")
    compiled = marshal.loads(args.builder_cache.read_bytes()[16:])
    compiled_main = next(value for value in compiled.co_consts
                         if hasattr(value, "co_name") and value.co_name == "main")
    compiled_numeric_vars = [name for name in compiled_main.co_varnames
                             if name in ("followers", "following", "tweet_count",
                                         "listed_count", "verified", "active_years")]
    archive_candidates = {}
    for path in args.manuscript_archives:
        with zipfile.ZipFile(path) as archive:
            archive_candidates[path.name] = [name for name in archive.namelist()
                                             if name.endswith(("clean_train_only_graph.pt",
                                                               "build_features_and_graph.py"))]
    result = {
        "scope": "Local TwiBot-20 HGT checkpoint input-contract audit",
        "official_train_zip_sha256": digest_file(args.official_zip),
        "official_train_member_sha256": official_train_hash,
        "manifest_train_sha256": manifest["input_sha256"],
        "official_train_matches_manifest": official_train_hash == manifest["input_sha256"],
        "local_qwen_weights_sha256": model_weights_hash,
        "manifest_qwen_weights_sha256": manifest["text_model_weights_sha256"],
        "local_qwen_matches_manifest": model_weights_hash == manifest["text_model_weights_sha256"],
        "qwen_hidden_width": text_width,
        "manifest_feature_width": int(manifest["feature_dim"]),
        "current_builder_numeric_width": builder_numeric_width,
        "current_builder_total_width": builder_width,
        "current_evaluator_numeric_width": evaluator_numeric_width,
        "current_evaluator_total_width": evaluator_width,
        "checkpoint_user_width": checkpoint_width,
        "checkpoint_sha256": digest_file(args.checkpoint),
        "saved_build_log_total_width": int(build_log_match.group(1)),
        "saved_build_log_numeric_width": int(build_log_match.group(4)),
        "saved_train_log_input_width": int(train_log_match.group(1)),
        "saved_graph_exists": args.graph.is_file(),
        "compiled_builder_cache_numeric_vars": compiled_numeric_vars,
        "manuscript_archive_recovery_candidates": archive_candidates,
        "ready_for_exact_frozen_inference": False,
        "reason": "The saved checkpoint and training log require 2054 inputs, whereas the current builder, evaluator and manifest specify 2053; the 2054-column graph artifact is absent. The missing sixth numeric input cannot be identified from these files.",
    }
    if not (result["official_train_matches_manifest"]
            and result["local_qwen_matches_manifest"]
            and builder_width == evaluator_width == manifest["feature_dim"]
            and checkpoint_width == int(train_log_match.group(1))
            == int(build_log_match.group(1))
            and checkpoint_width != builder_width
            and not args.graph.is_file()):
        raise ValueError("Observed input contract differs from the expected mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("official_train_zip_sha256",)}, indent=2))


if __name__ == "__main__":
    main()
