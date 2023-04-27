#!/bin/bash

# -----------------------------------------------------------------------------
#                               TO BE CUSTOMIZED
# -----------------------------------------------------------------------------
# Directory containing the source code
source scripts/env_vars.sh


main_bash_script="scripts/job.sh"

# SLURM options
slurm_log_dir="$HOME/slurm_logs"
notify_email="" # Leave empty ("") for no email
partition="long"

NUM_GPUS=1
export NUM_GPUS

# Resources, given number of GPUs requested
if [ "${partition}" = "main" ];
then
  # Set the maximum allowed on main
  mem=20
  cpus=4
  time=00:20:00
else
  mem=$(( $NUM_GPUS * 20 ))
  cpus=4
  time=00:20:00
fi

# The parameter of this function is the python arguments
submit_sbatch () {
    sbatch --job-name=lagrangian-slurm-%j.out \
        --time=$time \
        --cpus-per-task $cpus \
        --mem="$mem"G \
        --gres=gpu:$NUM_GPUS \
        --partition=$partition \
        --output=$slurm_log_dir/lagrangian-slurm-%j.out \
        --mail-type=ALL --mail-user=$notify_email \
        $main_bash_script $1
}


# -----------------------------------------------------------------------------

# model="logreg"
# optimizer="sgd"

model="mlp"
optimizer="adam"

# Config lists
declare -a _target_model=(30)
declare -a _target_layer=(15)
declare -a _lrs=(1e-4)

# declare -a _formulations=("qp")
# declare -a _penalty_gamma=(1.1)
# penalty_init=0.5

# declare -a _formulations=("exact")
# declare -a _penalty_gamma=(1.1)
# penalty_init=0.5

# declare -a _formulations=("lag")
# declare -a _lag_lr=(1e-6)

# declare -a _formulations=("alm")
# declare -a _penalty_gamma=(1.1)
# penalty_init=0.05

for model_target in "${_target_model[@]}"
do
  model_str="--config.train.target_model=$model_target"
  for layer_target in "${_target_layer[@]}"
  do
    layer_str="--config.train.target_layer=$layer_target"
    for lr in "${_lrs[@]}"
    do
        lr_str="--config.train.lr=$lr"
        for formulation in "${_formulations[@]}"
        do
            config_str="--config=configs/default.py:$model-$optimizer-$formulation"

            if [ "$formulation" = "qp" ];
            then
              for gamma in "${_penalty_gamma[@]}"
              do
                kwarg_str="--config.formulation.kwargs.penalty_gamma=$gamma --config.formulation.kwargs.penalty_coefficient=$penalty_init"
                submit_sbatch "$config_str $model_str $layer_str $lr_str $kwarg_str"
              done
            fi

            if [ "$formulation" = "exact" ];
            then
              for gamma in "${_penalty_gamma[@]}"
              do
                kwarg_str="--config.formulation.kwargs.penalty_gamma=$gamma --config.formulation.kwargs.penalty_coefficient=$penalty_init"
                submit_sbatch "$config_str $model_str $layer_str $lr_str $kwarg_str"
              done
            fi

            if [ "$formulation" = "lag" ];
            then
              for dual_lr in "${_lag_lr[@]}"
              do
                kwarg_str="--config.formulation.kwargs.dual_lr=$dual_lr"
                submit_sbatch "$config_str $model_str $layer_str $lr_str $kwarg_str"
              done
            fi

            if [ "$formulation" = "alm" ];
            then
              for gamma in "${_penalty_gamma[@]}"
              do
                kwarg_str="--config.formulation.kwargs.penalty_gamma=$gamma --config.formulation.kwargs.penalty_coefficient=$penalty_init"
                submit_sbatch "$config_str $model_str $layer_str $lr_str $kwarg_str"
              done
            fi

        done
    done
  done
done
