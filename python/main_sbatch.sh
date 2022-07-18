#!/bin/bash
#SBATCH -a 0-2
#SBATCH -c 24
#SBATCH -o log/main-%A-%a.log # %A=task array id, %a=task index number; if not doing array job (ie no -a) then replace %A%a with %j=job ID, 
#SBATCH -p all
#SBATCH -t 40
#SBATCH --mail-type ALL
#SBATCH --mail-user paul.m.krueger@gmail.com
module purge
# module load anacondapy\5.3.1 # probably python 2.7
module load anacondapy/2021.11
conda activate risky-choice
/usr/bin/time -av python3 main.py