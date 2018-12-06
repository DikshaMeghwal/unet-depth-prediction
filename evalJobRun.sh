#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=10GB
#SBATCH --mail-type=END
##SBATCH --mail-user=dm4511@nyu.edu
#!/bin/bash


PATH="~/anaconda3/bin/anaconda:$PATH"

python evaluate.py --data data --batch-size 5 $1
