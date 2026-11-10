@echo off
rem Usage: run_model.cmd gpt2-small|neo-125m   (resumable; appends to run_<model>.log)
cd /d "%~dp0.."
set PYTHONPATH=src
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_SYMLINKS=1
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
python "emnlp imp\run_estimates.py" --model %1 >> "emnlp imp\run_%1.log" 2>&1
