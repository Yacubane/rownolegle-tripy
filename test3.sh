#!/bin/bash -l

#SBATCH --nodes 1
#SBATCH --ntasks 24
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid-testing

module load plgrid/tools/python-intel/3.6.5 2>/dev/null
report_file=report-$(date --iso-8601=seconds).csv

test_iterations=5
thread_nums=(24 16 8 4 2 1)
stars_counts=(96)

echo "program;thread_num;stars_count;time" >> $report_file
for i in $(seq 1 $test_iterations); do
    for thread_num in "${thread_nums[@]}"; do
        for stars_count in "${stars_counts[@]}"; do
            for program in "./task1.py" "./task2.py"; do
                command="mpirun -np $thread_num $program $stars_count 0.05 1000"
                echo $command
                time=$($command)
                printf "%s;%d;%d;%lf\n" $program $thread_num $stars_count $time >> $report_file
            done
        done
    done
done
