#!/usr/bin/env bash
# The chain that ran on cml7 on 2026-09-29 at 858a799 (repeat study, then v5b_hard and v5b relabels with the
# found-lost-pair fix). Committed afterwards for provenance; it was first sent over ssh (daily-logs PLAN_repeat_study ledger).
cd /tmp2/chungyili/RoboLab
export CUDA_VISIBLE_DEVICES=0 PAIRS_DIR=output/test_lift/repeat_pairs
OBJECTS="banana dry_erase_marker hammer_2 hammer_4 hammer_5 ladle measuring_spoon remote_control snickers_bar soft_scrub spaghetti spatula_07 wooden_spoons" EXTRA_ARGS="--theta-profile hard --theta-seed 1" scripts/test_lift_repeat_sweep.sh output/test_lift/v5_repeat > /tmp2/chungyili/v5_repeat_sweep_long.log 2>&1
OBJECTS="apple_01 bagel_00 bin_a03 canned_tuna computer_mouse cream_cheese ketchup_bottle lychee01 measuring_cups_1 oatmeal_raisin_cookies plate_small red_block tomato_soup_can" EXTRA_ARGS="--theta-seed 0" scripts/test_lift_repeat_sweep.sh output/test_lift/v5_repeat > /tmp2/chungyili/v5_repeat_sweep_compact.log 2>&1
OBJECTS="banana dry_erase_marker hammer_2 hammer_4 hammer_5 ladle measuring_spoon remote_control snickers_bar soft_scrub spaghetti spatula_07 wooden_spoons" EXTRA_ARGS="--theta-profile hard --theta-seed 1" scripts/test_lift_v5_sweep.sh output/test_lift/v5b_hard > /tmp2/chungyili/v5b_hard_sweep.log 2>&1
scripts/test_lift_v5_sweep.sh output/test_lift/v5b > /tmp2/chungyili/v5b_sweep.log 2>&1
