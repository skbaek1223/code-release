"""Wrapper: run_re_guide.py with the multi-hop-friendly extractor prompt.

Same CLI as run_re_guide.py. Imports the patch first so by the time
run_re_guide imports get_extractor_instruction the patched version is in
place. Used for the 2WikiMQA compositional validation run only; the main
pipeline is not affected.
"""
import os
import sys

# Apply the prompt patch BEFORE run_re_guide imports get_extractor_instruction.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts_extractor_fix  # noqa: F401  (side effect: patches prompts.py)

# Now defer to the standard runner.
import run_re_guide  # noqa: E402

if __name__ == "__main__":
    run_re_guide.main()
