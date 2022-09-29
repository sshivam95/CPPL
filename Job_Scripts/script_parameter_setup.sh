declare -a instance_directory=("SWGCP_only_8")
declare -a solver_array=("cadical" )#"glucose")
declare -a time_out=(300)

for instance in "${instance_directory[@]}";
do
  for solver in "${solver_array[@]}";
  do
    for to in "${time_out[@]}";
    do
      sbatch run_script.sh $instance $solver $to
    done
  done
done
