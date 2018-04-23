

based on keras version 2.0.7
needs also pandas dataframe in addition to the "standard" packages that come with DeepJetCore 


Quick Manual
==============
 

To run domain adaptation on a local computer:

Models are defined in block_models.py and put together in AdaptMeDelphes.py.
The training is run with python AdaptMeDelphes.py <outdir> <run option>.
  The script has a help message listing the run options and other training parameters.


To run via scripts many in trainings in parralel on GPUs at CERN, follow the below:

log in CERN GPU:
ssh cms-gpu1080
refresh afs:
kinit
aklog

Look at which GPus are busy by: nvidia-smi

To run multiple trainings on the cmg-gpu machine, use the steer.sh script: steer.sh <output dir>. It needs to be adapted to your needs (what needs to be adapted?). Please also make sure the GPUs that you want to use are free (How?). These are defined in the beginning of the script (not tto clear?).
  
The steer script also runs the computation of averages (compute_averages.py) and the plotting (make_plots.py).
If training parameters are changed (in particular the number of trainings per run option to assess the statistical uncertainty), the plotting and possibly the averaging script needs to be adapted.


