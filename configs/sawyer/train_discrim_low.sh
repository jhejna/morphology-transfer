python scripts/train.py \
    --alg DSAC \
    --env Waypoint_Sawyer6Arm1 \
    --rand-init true \
    --xyz-skill true \
    --action-penalty 0.02 \
    --env-wrapper L2Low \
    --use-her true \
    --delta-max 0.04 0.04 0.01 0.2 0.2 0.2 \
    --epsilon 0.0225 \
    --sparse-reward 3.5 \
    --reward-scale 0.25 \
    --relative false \
    --survive-reward -0.5 \
    --goal-range-low -0.635 -0.635 -0.09 0.3 -0.35 -0.05 \
    --goal-range-high 0.635 0.635 0.09 0.95 0.35 0.25 \
    --additive-goals false \
    --learning-rate 0.0003 \
    --batch-size 128 \
    --buffer-size 1000000 \
    --layers 300 300 \
    --time-limit 100 \
    --timesteps 1200000 \
    --low-level 5DofArm1/Waypoint_Sawyer5Arm1_L2Low_SAC_0 \
    --discrim-relative false \
    --discrim-time-limit 80 \
    --discrim-decay true \
    --discrim-include-skill true \
    --discrim-learning-rate 0.0005 \
    --discrim-train-freq 6 \
    --discrim-stop 0.667
