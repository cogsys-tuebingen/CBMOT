from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import time
import copy

from nuscenes import NuScenes
from nuscenes.utils import splits
from tracker import PubTracker as Tracker
import fusion as fuse

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
    'car': 3,
    'truck': 4,
    'bus': 5.5,
    'trailer': 2,
    'pedestrian': 1,
    'motorcycle': 4,
    'bicycle': 2.5,
    'construction_vehicle': 1,
    'barrier': 1,
    'traffic_cone': 1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", type=str, default="work_dir", help="the dir to save logs and tracking results")
    parser.add_argument("--checkpoint", type=str, default='resources/infos_val_10sweeps_withvelo_filter_True.json')
    parser.add_argument("--hungarian", type=bool, default=False, help='use hungarian or greedy')
    parser.add_argument("--root", type=str, default="data/nuScenes")
    parser.add_argument("--version", type=str, default='v1.0-trainval')
    parser.add_argument("--max_age", type=int, default=40)
    parser.add_argument("--min_hits", type=int, default=1)
    parser.add_argument("--score_decay", type=float, default=0.0)
    parser.add_argument("--active_th", type=float, default=1.0)
    parser.add_argument("--deletion_th", type=float, default=0.0)
    parser.add_argument("--detection_th", type=float, default=0.0)
    parser.add_argument("--score_update", type=str, default=None)
    parser.add_argument("--Lidar_traindata", type=bool, default=False, help='gather score update function train data')
    parser.add_argument("--model_path", type=str, default=None)

    '''
    fusion specific parameters.
    '''
    parser.add_argument("--fusion", type=bool, default=False, help='use multimodal')
    parser.add_argument("--checkpoint2", type=str, help="file which contains the tracklets to fuse with")
    parser.add_argument("--decay1", type=float, default=0.2)
    parser.add_argument("--decay2", type=float, default=0.4)
    parser.add_argument("--star", type=bool, default=False)
    parser.add_argument("--del_th", type=float, default=0)

    '''
    velocity deviation specific variables. To include objects velocity in the similarity calculation of the 
    distance matrix before the matching algorithm is applied.
    For now: only applied on modality matching (fusion), not for detection-tracklet matching
    '''
    parser.add_argument("--v_min", type=float, default=0.4)
    parser.add_argument("--v_max", type=float, default=1.9)
    parser.add_argument("--v_weight", type=float, default=0.8)

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
    # added
    annotations = []
    for sample in nusc.sample:
        scene_name = nusc.get("scene", sample['scene_token'])['name']
        if scene_name not in scenes:
            continue

        timestamp = sample["timestamp"] * 1e-6
        token = sample["token"]
        frame = {'token': token, 'timestamp': timestamp}

        # start of a sequence
        if sample['prev'] == '':
            frame['first'] = True
            sample['first'] = True
        else:
            frame['first'] = False
            sample['first'] = False

        # added (detection_name is just named for easying the process)
        for annotation in reversed(sample['anns']):
            current = nusc.get("sample_annotation", annotation)
            current['detection_name'] = name_extraction(NUSCENES_TRACKING_NAMES, current['category_name'])

            if current['detection_name'] in NUSCENES_TRACKING_NAMES:
                current['label_preds'] = int(NUSCENES_TRACKING_NAMES.index(current['detection_name']))
                sample['anns'][sample['anns'].index(annotation)] = current
            else:
                sample['anns'].remove(annotation)

        sample_filtered = sample.copy()
        del sample_filtered['data']

        frames.append(frame)
        annotations.append(sample_filtered)

    del nusc

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, 'frames_meta.json'), "w") as f:
        json.dump({'frames': frames}, f)
    # added
    with open(os.path.join(args.work_dir, 'annotations.json'), "w") as f:
        json.dump({'samples': annotations}, f)


def main():
    args = parse_args()
    print('Deploy OK')

    # prepare tracker
    tracker = Tracker(max_age=args.max_age, hungarian=args.hungarian, noise=args.score_decay, active_th=args.active_th,
                      min_hits=args.min_hits, score_update=args.score_update, deletion_th=args.deletion_th,
                      detection_th=args.detection_th, model_path=args.model_path)

    # load first modality (resources)
    with open(args.checkpoint, 'rb') as f:
        predictions = json.load(f)['results']

    # prepare fusion between first and second modality
    if args.fusion:
        fusion = fuse.Fusion(hungarian=args.hungarian, decay1=args.decay1, decay2=args.decay2, star=args.star,
                             del_th=args.del_th, v_min=args.v_min, v_max=args.v_max, v_weight=args.v_weight)

        # load second modality (tracklets)
        with open(args.checkpoint2, 'rb') as f:
            tracklets2 = json.load(f)['results']

    # load frame meta
    with open(os.path.join(args.work_dir, 'frames_meta.json'), 'rb') as f:
        frames = json.load(f)['frames']

    # load annotations (ground truth) for train data
    with open(os.path.join(args.work_dir, 'annotations.json'), 'rb') as f:
        ground_truth = json.load(f)['samples']
    # prepare writen output file
    nusc_annos_trk = {
        "results": {},
        "meta": None,
    }
    nusc_annos_det = {
        "results": {},
        "meta": None,
    }

    # start tracking *****************************************
    print("Begin Tracking\n")
    start = time.time()
    size = len(frames)
    train_inputs = []
    train_labels = []
    # for each frame
    for i in range(size):

        # get frameID (=token)
        token = frames[i]['token']

        # reset tracking and id_log after one video sequence
        if frames[i]['first']:
            # use this for sanity check to ensure your token order is correct
            # print("reset ", i)
            tracker.reset()
            last_time_stamp = frames[i]['timestamp']

            if args.fusion:
                fusion.update_scene_velos(predictions, frames, i)  # get median velocities of current scene
                fusion.reset_id_log()  # reset id_log for the new scene

        # calculate time between two frames
        time_lag = (frames[i]['timestamp'] - last_time_stamp)
        last_time_stamp = frames[i]['timestamp']

        # resources of current frame (first modality)
        preds = predictions[token]

        # get tracklets of current frame (first modality)
        outputs = tracker.step_centertrack(preds, ground_truth[i], time_lag, args.version,
                                           args.Lidar_traindata)

        if args.Lidar_traindata and bool(outputs[1]):
            train_inputs += outputs[1]['input']
            train_labels += outputs[1]['labels']

        # if traindata true then output receive not the train_set
        if args.Lidar_traindata:
            outputs = outputs[0]

        # fuse first modality tracklets with second modality tracklets of current frame
        if args.fusion:
            tracks1 = copy.deepcopy(outputs)
            tracks2 = tracklets2[token]
            outputs = fusion.fuse(tracks1, tracks2)
        else:
            for item in outputs:
                item['tracking_score'] = item['detection_score']

        # prepare writen results file
        annos_trk = []
        annos_det = []

        for item in outputs:
            if 'active' in item and item['active'] < args.min_hits:
                continue

            nusc_det = {
                "sample_token": token,
                "translation": item['translation'],
                "size": item['size'],
                "rotation": item['rotation'],
                "velocity": item['velocity'],
                "tracking_id": str(item['tracking_id']),
                "detection_name": item['detection_name'],
                "detection_score": item['detection_score'],
                "attribute_name": item['attribute_name']
            }
            annos_det.append(nusc_det)

            if item['detection_name'] in NUSCENES_TRACKING_NAMES:
                nusc_trk = {
                    "sample_token": token,
                    "translation": item['translation'],
                    "size": item['size'],
                    "rotation": item['rotation'],
                    "velocity": item['velocity'],
                    "tracking_id": str(item['tracking_id']),
                    "tracking_name": item['detection_name'],
                    "tracking_score": item['tracking_score'],
                }
                annos_trk.append(nusc_trk)

        nusc_annos_trk["results"].update({token: annos_trk})
        nusc_annos_det["results"].update({token: annos_det})

    # calculate computation time
    end = time.time()
    second = (end - start)
    speed = size / second
    print("The speed is {} FPS".format(speed))

    # add meta info to writen result file
    if args.fusion:
        nusc_annos_trk["meta"] = {
            "use_camera": True,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
        nusc_annos_det["meta"] = {
            "use_camera": True,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
    else:
        nusc_annos_trk["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
        nusc_annos_det["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

    # get file location
    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    # save update score function train data
    if args.Lidar_traindata:
        train_path = 'train_data_2input.json'
        with open(os.path.join(args.work_dir, train_path), "w") as f:
            json.dump({'data': list(zip(train_inputs, train_labels))}, f)
    # write result file
    with open(os.path.join(args.work_dir, 'tracking_result.json'), "w") as f:
        json.dump(nusc_annos_trk, f)
    with open(os.path.join(args.work_dir, 'detection_result.json'), "w") as f:
        json.dump(nusc_annos_det, f)
    return speed


def eval_tracking():
    args = parse_args()
    eval(os.path.join(args.work_dir, 'tracking_result.json'),
         "val",
         args.work_dir,  # instead of args.work_dir,
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


def name_extraction(names_list, name_to_extract):
    for name in names_list:
        p = name_to_extract.find(name)
        if p != -1:
            np = p + len(name)
            return name_to_extract[p:np]


if __name__ == '__main__':
    save_first_frame()
    main()
    eval_tracking()
