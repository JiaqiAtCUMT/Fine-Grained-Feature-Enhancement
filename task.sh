#!/usr/bin/env bash
CONFIG=faster_rcnn_orpn_r50_fpn_1x_fair1m
CONFIG=faster_rcnn_orpn_r50_fpn_1x_fair1m_ms_al_n1
./test.sh work_dirs/$CONFIG/$CONFIG.py latest.pth

# CONFIG=faster_rcnn_orpn_r50_fpn_1x_fair1m_ms_al_n1
# # ./tools/dist_train.sh configs/obb/oriented_rcnn/$CONFIG.py 4
# ./test.sh work_dirs/$CONFIG/$CONFIG.py epoch_12.pth