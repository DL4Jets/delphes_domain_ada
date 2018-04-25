#! /bin/bash

#
# USAGE: steer.sh <output directory>
#
# probably this script needs adaptation if the scan should be performed
# in different variables than listed here by default.
# the structure, however, can be used in the same way and should
# be self-explanatory.
# For large scale trainings, refer to the comment "COMMENT FOR LARGE SCALE TRAININGS"
# a few lines below the user options

## steering options

# selects the number of the GPU that will be filled with jobs first
# inclusive - meaning lowestGpuToUse=5 will start with GPU 5!
lowestGpuToUse=5
# selects the number of the GPU that will be filled with jobs last; 
# inclusive - meaning highestGpuToUse=7 will also fill GPU 7 if needed!
highestGpuToUse=7


# list of different lambdas to scan
lmbs="30"
# list of different learning rates to scan
LRs="0.0005"
# list of different weights for the domain-adaptation loss to scan
weights="2"

# training option (run option in AdaptMeDelphes) that should be compared
# to the standard MC training or data training
compare='corrected_domain_adaptation'

# number of trainings to run for each configuration to estimate statistical
# fluctuations. if this parameter is changed, it will need to be changed in make_plot.py, too
ntrainings=5

# selects the number of processes that are run on a single GPU. 
# default: 5 is optimised for the default GPU fraction in AdaptMeDelphes.py.
# this parameter should not need changes
nprocpergpu=5


outputdir=$1
echo $outputdir

if [[  -z  $outputdir ]]
then
	echo "specify output dir"
	exit
fi
#next run with weight 2 but in same output dir


mkdir -p $outputdir

cp AdaptMeDelphes.py $outputdir/
cp make_samples.py $outputdir/
cp block_models.py $outputdir/
cp Layers.py $outputdir/
cp training_tools.py $outputdir/


ngpus=$(($highestGpuToUse-$lowestGpuToUse))
nprocstotal=$(($nprocpergpu*$ngpus))


nprocs=0
procpergpu=0
igpu=$lowestGpuToUse

# COMMENT FOR LARGE SCALE TRAININGS
# please refer to the occurence of these variables in case of large-scale trainings
# with constant data and MC training. Then it makes sense to just link the data/MC training
# output rather than repeating the training each time. The paths here are just an example
linkdata=/afs/cern.ch/user/j/jkiesele/work/DeepLearning/DomAdapt/delphes_domain_ada/stepwise3/0.0005/10/1/data_training
linkmc=/afs/cern.ch/user/j/jkiesele/work/DeepLearning/DomAdapt/delphes_domain_ada/stepwise3/0.0005/10/1/MC_training


for lmb in $lmbs; do
	for LR in $LRs; do
		for weight in $weights; do
			jobout=$outputdir/$LR/$lmb/$weight
			mkdir -p $jobout
			
			# uncomment if data and MC training should only be linked and comment the corresponding
			# parts in the following method loop in turn
			#ln -s $linkdata $jobout/
			#ln -s $linkmc $jobout/
		
			for method in $compare 'MC_training' 'data_training'; do	#'MC_training' 'data_training'	
				for i in $(seq $ntrainings); do
	   
	   				echo $jobout/$method/$i $method
	   
					python $outputdir/AdaptMeDelphes.py $jobout/$method/$i $method --weight $weight --lr $LR --lmb $lmb --gpu=$igpu --gpufraction=0.17 &> $jobout/$method.$i.log &
					
					echo "gpu ${igpu}, proc: ${procpergpu}"
					
					nprocs=$(($nprocs+1))
					procpergpu=$(($procpergpu+1))
					
					
				    if [ $nprocpergpu -eq $procpergpu ]
				    then
				    	if [ $igpu -eq $highestGpuToUse ]
				    	then
				    		igpu=$lowestGpuToUse
				    		echo waiting
				    		wait
				    	else
				    		igpu=$((igpu + 1))
				    	fi
				    	procpergpu=0
				    fi
				done
				#python compute_averages.py $jobout/$method		
			done
		done
	done
	#python make_plots.py $jobout
done
echo waiting
wait
for lmb in $lmbs; do
	for LR in $LRs; do
		for weight in $weights; do
			
			jobout=$outputdir/$LR/$lmb/$weight
			for method in 'MC_training' 'data_training' $compare; do
				python compute_averages.py $jobout/$method
			done
			python make_plots.py $jobout -c $compare
		done
	done
done

