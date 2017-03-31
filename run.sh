#!/usr/bin/env bash

###author : Qi Zhang ###
#######"""NJU"""########

file_path=/pandisk/2013_result/ctl/
chmod u+x follow_GFDL_tracking_mpi.py
sec_st_time=`date -d '2013-01-01 00:00:00 utc' +%s`
sec_ed_time=`date -d '2013-12-31 00:00:00 utc' +%s`
while [[ ${sec_st_time} -lt ${sec_ed_time} ]];
do
t=`date -d '1970-01-01 '${sec_st_time}' sec utc' -u +'%Y%m%d%H'`
YEAR=${t:0:4}
MONTH=${t:4:2}
DAY=${t:6:2}
HOUR=${t:8:2}
echo ${t}
time mpirun -n 5 ./follow_GFDL_tracking_mpi.py << EOF > path_${YEAR}${MONTH}${DAY}${HOUR}
${file_path}/wrfout_d01_${YEAR}-${MONTH}-${DAY}_${HOUR}:00:00.nc4
${YEAR}${MONTH}${DAY}${HOUR}
EOF
sec_st_time=$[${sec_st_time}+21600]
done
