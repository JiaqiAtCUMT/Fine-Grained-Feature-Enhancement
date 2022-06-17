#!/usr/bin/env bash
CONFIG=faster_rcnn_orpn_r50_fpn_1x_fair1m_ms_al_n1
# ./tools/dist_train.sh configs/obb/gliding_vertex/$CONFIG.py 4
./test.sh work_dirs/$CONFIG/$CONFIG.py latest.pth

# CONFIG=faster_rcnn_roitrans_r50_fpn_1x_fair1m
# ./tools/dist_train.sh configs/obb/roi_transformer/$CONFIG.py 4
# ./test.sh work_dirs/$CONFIG/$CONFIG.py latest.pth
