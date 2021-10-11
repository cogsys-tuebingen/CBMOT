from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import copy
import json
import math
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils import splits

from scipy.optimize import linear_sum_assignment as linear_assignment
from tracker import greedy_assignment


NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck',
]

# 99.9 percentile of the l2 velocity error distribution (per class / 0.5 second)
# This is an earlier statistics and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
NUSCENE_CLS_VELOCITY_ERROR = {
    'car': 4,
    'truck': 4,
    'bus': 5.5,
    'trailer': 3,
    'pedestrian': 1,
    'motorcycle': 13,
    'bicycle': 3,
    'construction_vehicle': 1,
    'barrier': 1,
    'traffic_cone': 1,
}


# compute distance matrix
def create_distance_matrix(tracks1, tracks2):
    # initialize distance matrix
    distances = np.ndarray(shape=(len(tracks1), len(tracks2)))

    # for every tracklet of both tracks lists
    for row in range(len(tracks1)):
        for col in range(len(tracks2)):

            # check if potential match has same class (else: invalid match)
            if (tracks1[row]['tracking_name'] == tracks2[col]['tracking_name']):

                # compute pure distance (eucl. distance)
                dist = math.sqrt((tracks1[row]['translation'][0] - tracks2[col]['translation'][0]) ** 2 + \
                                 (tracks1[row]['translation'][1] - tracks2[col]['translation'][1]) ** 2)

                # determine whether distance is close enough to count as a match
                if (dist <= NUSCENE_CLS_VELOCITY_ERROR[tracks1[row]['tracking_name']]):
                    distances[row][col] = dist

                # else: invalid match
                else:
                    distances[row][col] = 1e18

            # set distance to infinite if match is invalid
            else:
                distances[row][col] = 1e18

    return distances


def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results")
    parser.add_argument("--checkpoint", help="the dir to checkpoint which the model read from")
    parser.add_argument("--hungarian", type=bool, default=False)
    parser.add_argument("--root", type=str, default="data/nuScenes")
    parser.add_argument("--version", type=str, default='v1.0-trainval')

    args = parser.parse_args()

    return args


def save_first_frame():
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.root, verbose=True)
    if args.version == 'v1.0-trainval':
        scenes = splits.val
    elif args.version == 'v1.0-test':
        scenes = splits.test
    else:
        raise ValueError("unknown")

    frames = []
    for sample in nusc.sample:
        scene_name = nusc.get("scene", sample['scene_token'])['name']
        if scene_name not in scenes:
            continue

        timestamp = sample["timestamp"] * 1e-6
        token = sample["token"]
        frame = {}
        frame['token'] = token
        frame['timestamp'] = timestamp

        # start of a sequence
        if sample['prev'] == '':
            frame['first'] = True
        else:
            frame['first'] = False
        frames.append(frame)

    del nusc

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, 'frames_meta.json'), "w") as f:
        json.dump({'frames': frames}, f)


def main():
    args = parse_args()
    print('Deploy OK')

    # load CenterTrack trackings
    with open(args.checkpoint, 'rb') as f:
        ct_tracking = json.load(f)['results']

    # read frame meta
    with open(os.path.join(args.work_dir, 'frames_meta.json'), 'rb') as f:
        frames=json.load(f)['frames']

    # prepare writen output file
    nusc_annos = {
        "results": {},
        "meta": None,
    }

    # start tracking id rearrangement ************************
    tracking_id_counter = 1
    previous_token = -1

    # for each frame
    for i in range(len(ct_tracking)):
        token = frames[i]['token']  # get frameID (=token)

        # reset tracking after one video sequence
        if frames[i]['first']:
            # tracker.reset()
            tracking_id_counter = 1

            # in the first frame, simply give every tracklet a unique tracking id in ascending order
            for tracklet in ct_tracking[token]:
                tracklet['tracking_id'] = tracking_id_counter
                tracking_id_counter += 1

        # for all subsequent frames, do matching with its previous frame
        else:
            # calculate distance matrix between current tracklets and tracklets from previous frame
            track_distances = create_distance_matrix(ct_tracking[token], ct_tracking[previous_token])

            # find best matching of the two tracking results:
            if args.hungarian:

                # use hungarian algorithm to find best matching tracklets
                matches_temp1 = linear_assignment(copy.deepcopy(track_distances))

                # reshape hungerians output
                matches_temp2 = []
                for i in range(len(matches_temp1[0])):
                    matches_temp2.append([matches_temp1[0][i], matches_temp1[1][i]])

                # get rid of matches with infinite distance
                matched_ids = []
                for m in matches_temp2:
                    if track_distances[m[0], m[1]] < 1e16:
                        matched_ids.append(m)
                matched_ids = np.array(matched_ids).reshape(-1, 2)

            else:
                # use greedy algorithm
                matched_ids = greedy_assignment(copy.deepcopy(track_distances))

            # assign new tracking ids:
            for i in range(len(ct_tracking[token])):

                # for all matched ids, assign previous tracking id
                if i in matched_ids[:,0]:
                    matching_id = matched_ids[[item[0] == i for item in matched_ids].index(True)][1]
                    ct_tracking[token][i]['tracking_id'] = ct_tracking[previous_token][matching_id]['tracking_id']

                # for all unmatched tracklets, give new tracking ids
                else:
                    ct_tracking[token][i]['tracking_id'] = tracking_id_counter
                    tracking_id_counter += 1

        previous_token = token

        # prepare writen results file
        annos = []
        for item in ct_tracking[token]:
            nusc_anno = {
                "sample_token": token,
                "translation": item['translation'],
                "size": item['size'],
                "rotation": item['rotation'],
                "velocity": item['velocity'],
                "detection_name": item['detection_name'],
                "attribute_name": item['attribute_name'],
                "detection_score": item['detection_score'],
                "tracking_name": item['tracking_name'],
                "tracking_score": item['tracking_score'],
                "tracking_id": item['tracking_id'],
            }
            annos.append(nusc_anno)
        nusc_annos["results"].update({token: annos})

    # add meta info to writen result file
    nusc_annos["meta"] = {
        "use_camera": True,
        "use_lidar": False,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    # get file location
    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # write result file
    with open(os.path.join(args.work_dir, 'centertrack_tracks.json'), "w") as f:
        json.dump(nusc_annos, f)
    return 0


def eval_tracking():
    args = parse_args()
    eval(os.path.join(args.work_dir, 'centertrack_tracks.json'),
        "val",
        args.work_dir,
        args.root
    )


def eval(res_path, eval_set="val", output_dir=None, root_path=None):
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory as track_configs

    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=root_path,
    )
    metrics_summary = nusc_eval.main()


if __name__ == '__main__':
    save_first_frame()
    main()
    eval_tracking()
