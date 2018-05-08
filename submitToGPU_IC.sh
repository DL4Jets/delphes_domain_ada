#!/bin/bash
#$ -cwd
#$ -q gpu.q
#$ -v OMP_NUM_THREADS=16
#$ -l h_rt=24:0:0 
#$ -l gpu=1 #more is not yet possible for some reason

source /vols/build/cms/amartell/MLTag/XTagger/Env/env_gpu.sh 
source /vols/build/cms/mkomm/cuDNNv7_cuda8/setupCuDNN.sh

cd /vols/build/cms/amartell/DL4J/delphes_domain_ada

baseINPUTDIR=/vols/cms/amartell/DNN_domainAda_testGPU
mkdir -p $baseINPUTDIR

INPUTDIR=$baseINPUTDIR/MC_training
mkdir -p $INPUTDIR
python AdaptMeDelphes.py $INPUTDIR/1 MC_training -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/1.log &
python AdaptMeDelphes.py $INPUTDIR/2 MC_training -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/2.log &
python AdaptMeDelphes.py $INPUTDIR/3 MC_training -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/3.log &
python AdaptMeDelphes.py $INPUTDIR/4 MC_training -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/4.log &
python AdaptMeDelphes.py $INPUTDIR/5 MC_training -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/5.log &

wait

INPUTDIR=$baseINPUTDIR/data_training
mkdir -p $INPUTDIR
python AdaptMeDelphes.py $INPUTDIR/1 data_training -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/1.log &
python AdaptMeDelphes.py $INPUTDIR/2 data_training -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/2.log &
python AdaptMeDelphes.py $INPUTDIR/3 data_training -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/3.log &
python AdaptMeDelphes.py $INPUTDIR/4 data_training -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/4.log &
python AdaptMeDelphes.py $INPUTDIR/5 data_training -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/5.log &

wait

INPUTDIR=$baseINPUTDIR/stepwise_domain_adaptation
mkdir -p $INPUTDIR
python AdaptMeDelphes.py $INPUTDIR/1 stepwise_domain_adaptation -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/1.log &
python AdaptMeDelphes.py $INPUTDIR/2 stepwise_domain_adaptation -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/2.log &
python AdaptMeDelphes.py $INPUTDIR/3 stepwise_domain_adaptation -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/3.log &
python AdaptMeDelphes.py $INPUTDIR/4 stepwise_domain_adaptation -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/4.log &
python AdaptMeDelphes.py $INPUTDIR/5 stepwise_domain_adaptation -i='/vols/cms/amartell/DelphesTrainingDataset/numpyx2/' --gpufraction=0.17 --runIC &> $INPUTDIR/5.log &


wait


python compute_averages.py $baseINPUTDIR/MC_training
wait
python compute_averages.py $baseINPUTDIR/data_training
wait
python compute_averages.py $baseINPUTDIR/stepwise_domain_adaptation
wait


python make_plots.py $baseINPUTDIR -c stepwise_domain_adaptation
