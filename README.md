# BioEmu benchmark

job_8gpu.sh:
Each Athena node has 8 NVIDIA A100 cards. BioEmu does not natively support multi-GPU via DataParallel, so we run 8 parallel processes - each on a single GPU.
