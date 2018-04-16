#! /bin/bash


#inclusive!
lowgpu=5
#inclusive!
maxgpu=7

ntrainings=5
nprocpergpu=5

outputdir=$1
echo $outputdir

if [[  -z  $outputdir ]]
then
	echo "specify output dir"
	exit
fi
lmbs="10"
LRs="0.002"
weights="0.1 1"
#next run with weight 2 but in same output dir


mkdir -p $outputdir

cp AdaptMeDelpes.py $outputdir/
cp make_samples.py $outputdir/
cp models.py $outputdir/
cp Layers.py $outputdir/

ngpus=$(($maxgpu-$lowgpu))
nprocstotal=$(($nprocpergpu*$ngpus))


nprocs=0
procpergpu=0
igpu=$lowgpu



for lmb in $lmbs; do
	for LR in $LRs; do
		for weight in $weights; do
			jobout=$outputdir/$LR/$lmb/$weight
		
			for method in 'pretrained_domain_adaptation' 'MC_training' 'data_training'; do	#'MC_training' 'data_training'	
				for i in $(seq $ntrainings); do
	   
	   				echo $jobout/$method/$i $method
	   
				    mkdir -p $jobout
					python $outputdir/AdaptMeDelpes.py $jobout/$method/$i $method --weight $weight --lr $LR --lmb $lmb --gpu=$igpu --gpufraction=0.17 &> $jobout/$method.$i.log &
					
					echo "gpu ${igpu}, proc: ${procpergpu}"
					
					nprocs=$(($nprocs+1))
					procpergpu=$(($procpergpu+1))
					
					
				    if [ $nprocpergpu -eq $procpergpu ]
				    then
				    	if [ $igpu -eq $maxgpu ]
				    	then
				    		igpu=$lowgpu
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
			for method in 'MC_training' 'data_training' 'pretrained_domain_adaptation'; do
				python compute_averages.py $jobout/$method
			done
			python make_plots.py $jobout
		done
	done
done

