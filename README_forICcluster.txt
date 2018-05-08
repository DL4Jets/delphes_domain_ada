to setup the environment on the IC cluster and run a training with GPUs

1) get the package in https://github.com/LLPDNNX/XTagger
and follow the readme for the installation
    - NB. for KERAS 2.0.7 and tensorflow 1.0.1, configure the XTagger/Env/packages_cpu.pip and XTagger/Env/packages_gpu.pip 
    - follow instructions in the rEADME to source the cpu / gpu environment
    - also install pandas in the cpu and gpu environment
    

2) get the package in https://github.com/DL4Jets/delphes_domain_ada
example of script to submit the parallel training on GPUs is submitToGPU_IC.sh
via: qsub submitToGPU_IC.sh


3) EXTRAS:
   - to check the nodes with GPUs: qstat -f 
   - logging on the GPU node allows to check the status of the GPU: nvidia-smi
   - jobs cannot be submit from the GPU node
 
