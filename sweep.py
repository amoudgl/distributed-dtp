"""
This script iterates over all combinations of hyper-parameters provided in sweep config 
and submits batch job for each experiment to ComputeCanada cluster.
"""
import itertools
import subprocess

if __name__ == "__main__":
    # Meta info
    account = "def-eugenium"
    user = "amoudgl"
    network = "simple_vgg"
    algo = "dtp"
    time = "90:0:0"

    # Hyper params to sweep over
    sweep_config = [
        [            
            "--batch_size 256",
        ],
        [
            "--num_workers 4",
        ],
        [
            "--dataset imagenet32",
        ],        
        [
            "--seed 124",
            "--seed 125",
            "--seed 126",
            "--seed 127",
        ],
        [
            "--feedback_training_iterations 25 35 40 60 25",
        ],
        [
            "--f_optim.lr 0.01 cosine",
            "--f_optim.lr 0.05 cosine",
            "--f_optim.lr 0.01 step --step_size 45",
            "--f_optim.lr 0.05 step --step_size 45",
        ],
    ]
    init_commands = f"module load python/3.8 && source /scratch/{user}/py38/bin/activate && cd /scratch/{user}/scalingDTP && export WANDB_MODE=offline"
    python_command = f"python main_pl.py run {algo} {network}"
    sbatch_command = f"sbatch --gres=gpu:1 --account={account} --time={time} --cpus-per-task=16 --mem=48G"

    # Submit batch jobs for all combinations
    all_args = list(itertools.product(*sweep_config))
    print(f"Total jobs = {len(all_args)}")
    for args in all_args:
        args = " ".join(args)
        job_command = (
            sbatch_command
            + ' --wrap="'
            + init_commands
            + " && "
            + python_command
            + " "
            + args
            + '"'
        )
        print(job_command)
        subprocess.run(job_command, shell=True)
