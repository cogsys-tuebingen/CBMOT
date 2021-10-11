from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import copy
import numpy as np
import itertools

from scipy.optimize import linear_sum_assignment as linear_assignment

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


# Euclidean distance in 1D *****************************************************
def eucl1D(a, b):
    return math.sqrt(a ** 2 + b ** 2)


# Euclidean distance in 2D *****************************************************
def eucl2D(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# greedy matching algorithm *****************************************************
def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)


# compute distance matrix *****************************************************
def create_distance_matrix(tracks1, tracks2, velos, velo_dev):
    # extract centerpoints
    centerpoints_cp = np.array([item['ct'] for item in tracks1])
    velos_cp = np.array([eucl1D(item['velocity'][0], item['velocity'][1]) for item in tracks1])
    centerpoints_ct = np.array([item['translation'][:2] for item in tracks2])

    # initialize distance matrix
    distances = np.ndarray(shape=(len(centerpoints_cp), len(centerpoints_ct)))

    # compute distances
    row = -1
    for point_cp in tracks1:
        col = -1
        row += 1
        for point_ct in tracks2:
            col += 1

            # check if potential match has same class (else: invalid match)
            if point_cp['detection_name'] == point_ct['tracking_name']:

                # compute pure distance (eucl. distance)
                dist = eucl2D(centerpoints_cp[row], centerpoints_ct[col])

                # objects deviation in velocity against the class specific mean velocity (in percentage)
                median_vel = velos[point_cp['detection_name']][point_cp['attribute_name']]
                if median_vel == 0:
                    vel_dev = 1 if velos_cp[row] == 0 else velo_dev['max']
                else:
                    vel_dev = np.clip((velos_cp[row] / median_vel) ** velo_dev['weight'],
                                      a_min=velo_dev['min'], a_max=velo_dev['max'])

                # determine whether distance is close enough to count as a match
                if dist <= NUSCENE_CLS_VELOCITY_ERROR[point_cp['detection_name']] * vel_dev:
                    distances[row][col] = dist

                # else: invalid match
                else:
                    distances[row][col] = math.inf

            # set distance to infinite if match is invalid
            else:
                distances[row][col] = math.inf

    return distances


# class to fuse two sets of tracklets *****************************************************
class Fusion(object):
    def __init__(self, hungarian=False, decay1=0.2, decay2=0.2, star=True, del_th=0, v_min=1, v_max=1, v_weight=0):

        # fusion parameters
        self.hungarian = hungarian
        self.decay1 = decay1
        self.decay2 = decay2
        self.star = star        # if True: score decay is only applied if unmatched
        self.del_th = del_th

        # velocity parameters (to consider velocities in the objects distances)
        self.v_dev = {'min': v_min, 'max': v_max, 'weight': v_weight}
        self.velos = {}

        # id management (to map to a common tracking_id over all tracking result sources)
        # id_log: 'tracking source': [original tracking_id, new tracking_id, birthdate of the new tracking_id]
        self.id_log = {'set1': np.zeros(shape=(2, 3)) - 1, 'set2': np.zeros(shape=(2, 3)) - 1}
        self.id_counter = 1
        self.frame_id = 0

    # reset id management at the beginning of a new scene *****************************************************
    def reset_id_log(self):
        self.id_log = {'set1': np.zeros(shape=(2, 3)) - 1, 'set2': np.zeros(shape=(2, 3)) - 1}
        self.id_counter = 1
        self.frame_id = 0

    # get median velocities for current scene *****************************************************
    def update_scene_velos(self, predictions, frames, i):
        velos = {}

        while True:
            # LIDAR detections of current frame
            preds = predictions[frames[i]['token']]

            for item in preds:
                # initialize new class specific velocity array (if it not already exist)
                if item['detection_name'] not in velos:
                    velos[item['detection_name']] = {}

                if item['attribute_name'] not in velos[item['detection_name']]:
                    velos[item['detection_name']][item['attribute_name']] = np.array([])

                # calculate velocity:
                velocity = eucl1D(item['velocity'][0], item['velocity'][1])

                # append velocity to the list of velocities:
                velos[item['detection_name']][item['attribute_name']] = np.append(
                    velos[item['detection_name']][item['attribute_name']], velocity)

            # next frame
            i += 1

            # stop if next scene starts (or no more frames exist)
            if i >= len(frames) or frames[i]['first']:
                break

        # median for every object separately:
        for classes in velos:
            for attribute in velos[classes]:
                velos[classes][attribute] = np.median(velos[classes][attribute])

        self.velos = velos

    # fuse two different sets of tracklets *****************************************************
    def fuse(self, tracks1, tracks2):
        self.frame_id += 1

        # create a distance matrix of both tracking results
        # include invalid matches into the distance matrix (distance > max distance and different types)
        track_distances = create_distance_matrix(tracks1, tracks2, self.velos, self.v_dev)
        track_distances[track_distances > 1e18] = 1e18  # invalid matches get a very high number

        # find best matching of the two tracking results:
        if self.hungarian:

            # use hungarian algorithm to find best matching tracklets
            matches_temp = linear_assignment(copy.deepcopy(track_distances))

            # get rid of matches with infinite distance
            matched_ids = []
            for m in matches_temp:
                if track_distances[m[0], m[1]] < 1e16:
                    matched_ids.append(m)
            matched_ids = np.array(matched_ids).reshape(-1, 2)

        else:
            # use greedy algorithm
            matched_ids = greedy_assignment(copy.deepcopy(track_distances))

        # initialize joined output
        outputs = []

        # go through all FIRST MODALITY tracklets
        for tracklet_id in range(len(tracks1)):
            tracklet1 = tracks1[tracklet_id]
            tracklet = copy.deepcopy(tracklet1)

            # MATCHED TRACKLETS:
            if tracklet_id in matched_ids[:, 0]:
                tracklet2 = tracks2[matched_ids[list(itertools.chain(matched_ids[:, 0] == tracklet_id)).index(True)][1]]

                # star: only apply score decay if unmatched
                if not self.star:
                    tracklet1['detection_score'] -= self.decay1
                    tracklet2['tracking_score'] -= self.decay2

                # apply score update function (use eq. 6 [multiplication])
                tracklet['tracking_score'] = 1 - (1 - tracklet1['detection_score']) * (1 - tracklet2['tracking_score'])

                # if score is too low (below deletion threshold), dump this tracklet
                # (with current update function, the score of matched tracklets should always grow,
                # but with different update functions it might happen)
                if tracklet['tracking_score'] < self.del_th:
                    continue

                # update tracking id ***

                # check, whether tracking ids are already inside the id log structure
                log_flag1 = tracklet1['tracking_id'] in self.id_log['set1'][:, 0]
                log_flag2 = tracklet2['tracking_id'] in self.id_log['set2'][:, 0]

                # copy id log values if already contained in id_log
                if log_flag1:
                    log_val1 = copy.deepcopy(self.id_log['set1'][self.id_log['set1'][:, 0] == tracklet1['tracking_id']])[0]
                if log_flag2:
                    log_val2 = copy.deepcopy(self.id_log['set2'][self.id_log['set2'][:, 0] == tracklet2['tracking_id']])[0]

                # if both tracking ids are new, crete a new common tracking id
                if not log_flag1 and not log_flag2:
                    self.id_log['set1'] = np.vstack([self.id_log['set1'], [tracklet1['tracking_id'], self.id_counter, self.frame_id]])
                    self.id_log['set2'] = np.vstack([self.id_log['set2'], [tracklet2['tracking_id'], self.id_counter, self.frame_id]])
                    self.id_counter += 1

                # if ct tracking id is unknown, copy the tracking id from cp
                elif log_flag1 and not log_flag2:
                    self.id_log['set2'] = np.vstack(
                        [self.id_log['set2'], [tracklet2['tracking_id'], log_val1[1], log_val1[2]]])

                # if cp tracking id is unknown, copy the tracking id from ct
                elif not log_flag1 and log_flag2:
                    self.id_log['set1'] = np.vstack(
                        [self.id_log['set1'], [tracklet1['tracking_id'], log_val2[1], log_val2[2]]])

                # if both tracking ids are already known (used before), ...
                elif log_flag1 and log_flag2:
                    if log_val1[1] != log_val2[1]:  # ... and if not equal ...
                        if log_val1[2] <= log_val2[2]:  # ... take the older tracking id (overwrite the younger one)
                            self.id_log['set2'][self.id_log['set2'][:, 0] == tracklet2['tracking_id']][0][1] = log_val1[1]
                            self.id_log['set2'][self.id_log['set2'][:, 0] == tracklet2['tracking_id']][0][2] = log_val1[2]
                        else:
                            self.id_log['set1'][self.id_log['set1'][:, 0] == tracklet1['tracking_id']][0][1] = log_val2[1]
                            self.id_log['set1'][self.id_log['set1'][:, 0] == tracklet1['tracking_id']][0][2] = log_val2[2]

                # save the (new) common tracking id
                tracklet['tracking_id'] = self.id_log['set1'][self.id_log['set1'][:, 0] == tracklet['tracking_id']][0][1]

                # add current tracklet to the tracklets list
                outputs.append(tracklet)

            # UNMATCHED TRACKLETS of first modality:
            else:
                # apply score decay
                tracklet['tracking_score'] = tracklet['detection_score'] - self.decay1

                # if score is too low (below deletion threshold), dump this tracklet
                if tracklet['tracking_score'] < self.del_th:
                    continue

                # if tracking id is not known yet, give new id (else: don't change its id)
                if tracklet['tracking_id'] not in self.id_log['set1'][:, 0]:
                    self.id_log['set1'] = np.vstack([self.id_log['set1'], [tracklet['tracking_id'], self.id_counter, self.frame_id]])
                    self.id_counter += 1

                # save the (new) tracking id
                tracklet['tracking_id'] = self.id_log['set1'][self.id_log['set1'][:, 0] == tracklet['tracking_id']][0][1]

                # add current tracklet to the tracklets list
                outputs.append(tracklet)

        # go through all SECOND MODALITY tracklets
        for tracklet_id in range(len(tracks2)):

            # UNMATCHED TRACKLETS of second modality
            if tracklet_id not in matched_ids[:, 1]:    # (simply ignore all matches, since we have them already)
                tracklet = copy.deepcopy(tracks2[tracklet_id])

                # apply score decay
                tracklet['tracking_score'] = tracklet['tracking_score'] - self.decay2

                # if score is too low (below deletion threshold), dump this tracklet
                if tracklet['tracking_score'] < self.del_th:
                    continue

                # if tracking id is not known yet, give new id (else: don't change its id)
                if tracklet['tracking_id'] not in self.id_log['set2'][:, 0]:
                    self.id_log['set2'] = np.vstack([self.id_log['set2'], [tracklet['tracking_id'], self.id_counter, self.frame_id]])
                    self.id_counter += 1

                # save the (new) tracking id
                tracklet['tracking_id'] = self.id_log['set2'][self.id_log['set2'][:, 0] == tracklet['tracking_id']][0][1]

                # add current tracklet to the tracklets list
                outputs.append(tracklet)

        return outputs