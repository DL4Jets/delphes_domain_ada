

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
ssh cmg-gpu1080
refresh afs:
kinit
aklog

Look at which GPus are busy by: nvidia-smi

To run multiple trainings on the cmg-gpu machine, use the steer.sh script: steer.sh <output dir>. It needs to be adapted to your needs (what needs to be adapted?). Please also make sure the GPUs that you want to use are free. More information can be found in the script itself.
  
The steer script also runs the computation of averages (compute_averages.py) and the plotting (make_plots.py).
If training parameters are changed (in particular the number of trainings per run option to assess the statistical uncertainty), the plotting and possibly the averaging script needs to be adapted.

Quickstart and minimal workflow
================================

create multiple trainings with the same configuration

```
indir=SOME_DIRECTORY_NAME
mkdir -p $indir
python AdaptMeDelphes.py $indir/1 MC_training --gpu=7 --gpufraction=0.17 OPTIONS &> $indir/1.log &
python AdaptMeDelphes.py $indir/2 MC_training --gpu=7 --gpufraction=0.17 OPTIONS &> $indir/2.log &
python AdaptMeDelphes.py $indir/3 MC_training --gpu=7 --gpufraction=0.17 OPTIONS &> $indir/3.log &
python AdaptMeDelphes.py $indir/4 MC_training --gpu=7 --gpufraction=0.17 OPTIONS &> $indir/4.log &
python AdaptMeDelphes.py $indir/5 MC_training --gpu=7 --gpufraction=0.17 OPTIONS &> $indir/5.log &

wait
```

The output will be something like this

```
>>> ls SOME_DIRECTORY_NAME
1  1.log  2  2.log  3  3.log  4  4.log  5  5.log
>>> ls SOME_DIRECTORY_NAME/1
history.npy  predictions.npy
```

Once done you need to compute the averages

```
python compute_averages.py SOME_DIRECTORY_NAME
```

Now make some plots! This part currently assumes that there are three directories in the same directory INPUTDIR:
   * data_training - data only training 
   * MC_training - MC only training
   * SOME_DIRECTORY_NAME - anything you want to compare

To make plots:
```
python make_plots.py INPUTDIR -c SOME_DIRECTORY_NAME
```

And you will get in INPUTDIR the plots

