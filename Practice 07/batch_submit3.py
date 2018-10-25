import os
import time

num_layer = 1
#dim_list = [256]
#lr_list = [0.01]
#rr_list = [0.001]
dim_list = [128, 256]
lr_list = [0.001, 0.01]
rr_list = [0.01, 0.001, 0.0001]

count = 0
for hidden_dim in dim_list:
    for lr in lr_list:
        for rr in rr_list:
            count += 1
            job_name = 'smiles_rnn_logP3_'+str(hidden_dim) + '_' + str(lr) + '_' + str(rr)
            file_name = job_name+'.out' 

            f=open('test-batch'+str(count)+'.sh','w')
            f.write('''#!/bin/bash
#PBS -N '''+job_name+'''
#PBS -l nodes=1:ppn=7
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
echo 'cat $PBS_NODEFILE'
cat $PBS_NODEFILE
NPROCS='wc -l < $PBS_NODEFILE'

for DEVICEID in 'seq 0 3'; do
    AVAILABLE='nvidia-smi -i ${DEVICEID} | grep "No" | wc -l'
    if [ ${AVAILABLE} == 1 ] ; then
        break;
    fi

done
date
source ~/.bashrc
source activate jaechang-python-3.6
echo $DEVICEID
export CUDA_VISIBLE_DEVICES=$DEVICEID
export OMP_NUM_THREADS=1

python -u smiles_rnn_logP3.py ''' + str(num_layer) + ' ' + str(hidden_dim) + ' ' + str(lr) + ' ' + str(rr) + ' ' + '> results/'''+file_name+'''

date''')
            f.close()
            os.system('qsub test-batch'+str(count)+'.sh')
            print (num_layer, hidden_dim, lr, rr)
            time.sleep(0.5)
