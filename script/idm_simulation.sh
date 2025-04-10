cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

export NUPLAN_DATA_ROOT="/data/workspace/zhangjunrui/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/data/workspace/zhangjunrui/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="exp/idm_planner"
export PYTHONPATH=$PYTHONPATH:$(pwd)

PLANNER=pluto_planner
BUILDER=nuplan_challenge
FILTER=test14-hard
# CKPT=$4
# VIDEO_SAVE_DIR=$5

# CHALLENGE="closed_loop_nonreactive_agents"
CHALLENGE="closed_loop_reactive_agents"
# CHALLENGE="open_loop_boxes"

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    scenario_builder=$BUILDER \
    scenario_filter=$FILTER \
    worker.threads_per_node=40 \
    verbose=true \
    experiment_uid="pluto_planner/$FILTER" \



