#!/usr/bin/env bash

###author : Qi Zhang ###
#######"""NJU"""########

file_names=`ls bwp*.dat`

for file in ${file_names}:
do
echo ${file}
cat ${file} | while read line
do
mpirun -n 5 follow_GFDL_tracking_mpi.py << EOF
${line}
/pandisk/2013_result/no_restart_cfsr_satinfo
:00:00.nc4
EOF
done
done
