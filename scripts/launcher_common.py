"""Argv-patching helpers for run_search_o1_wiki.py's --preset support.

run_search_o1_wiki.py must set CUDA_VISIBLE_DEVICES from --gpus/--gpu_groups
before it imports torch/vllm, so its --preset resolution has to mutate
sys.argv up front (before argparse ever runs), rather than post-process a
parsed Namespace. These helpers implement that: inject preset flag defaults
into sys.argv unless already present, and split a single --gpus list into
one-GPU-per-dataset --parallel groups. Any flag the user already passed on
the command line takes precedence.

run_all_datasets.py doesn't have that early CUDA-visible-devices constraint,
so it resolves --preset directly on the parsed argparse Namespace instead
and doesn't use this module.
"""
import sys


def has_flag(flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])


def flag_value(flag: str):
    argv = sys.argv[1:]
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    for a in argv:
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def remove_flag(flag: str):
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == flag:
            i += 2
            continue
        if a.startswith(flag + "="):
            i += 1
            continue
        new_argv.append(a)
        i += 1
    sys.argv[:] = new_argv


def apply_preset(preset: dict):
    """Inject preset flag defaults into sys.argv unless already present."""
    for flag, val in preset.items():
        if not has_flag(flag):
            sys.argv += [flag, val]


def auto_parallelize_gpus():
    """Split a single --gpus list into one-GPU-per-dataset --parallel groups
    (e.g. --gpus 6,7 -> --parallel --gpu_groups "6;7"), so each GPU runs an
    independent worker (TP=1) and the launcher dispatches pending datasets
    onto whichever GPU frees up next. Skipped if --no_auto_parallel,
    --parallel, or --gpu_groups was passed explicitly.
    """
    disable = has_flag("--no_auto_parallel")
    if disable:
        remove_flag("--no_auto_parallel")
    if (not disable and has_flag("--gpus")
            and not has_flag("--parallel") and not has_flag("--gpu_groups")):
        gpu_list = [g.strip() for g in (flag_value("--gpus") or "").split(",") if g.strip()]
        if len(gpu_list) >= 2:
            remove_flag("--gpus")
            sys.argv += ["--parallel", "--gpu_groups", ";".join(gpu_list)]
