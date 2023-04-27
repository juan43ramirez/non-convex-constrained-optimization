export SRC_DIR=$HOME/repos/lagrangian-like-optimization
export LOG_DIR=$HOME/slurm_logs

export WANDB_ENTITY=juan_ramirez_research
export WANDB_PROJECT=lagrangian
export WANDB_DIR=/repos/lagrangian-like-optimization/wandb

# Automatically set SLURM_TMPDIR when running on interactive sessions
if [ "$SLURM_TMPDIR" = "" ]; then
    export SLURM_TMPDIR=/Tmp/slurm.${SLURM_JOB_ID}.0
fi
