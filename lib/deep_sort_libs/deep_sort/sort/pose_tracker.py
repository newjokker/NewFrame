# vim: expandtab:ts=4:sw=4
import numpy as np
from collections import deque

from .track import TrackState, Track
from .detection import Detection
from .tracker import Tracker


class pDetection(Detection):
    def __init__(self, tlwh, keypoints, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.keypoints = keypoints
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)


class pTrack(Track):
    def __init__(self, mean, covariance, track_id, n_init, max_age=30, 
                 buffer=30, features=None):
        super(pTrack, self).__init__(mean, covariance, track_id, n_init, max_age,
                features) 

        # keypoints list for use in Actions prediction.
        self.keypoints_list = deque(maxlen=buffer)

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(self.mean, self.covariance,
                                               detection.to_xyah())
        self.features.append(detection.feature)
        self.keypoints_list.append(detection.keypoints)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

class pTracker(Tracker):
    def __init__(self, matrix, max_iou_distance=0.7, max_age=70, n_init=3, buffer=30):
        super(pTracker, self).__init__(matrix, max_iou_distance, max_age, n_init)
        self.buffer = buffer

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(pTrack(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            self.buffer, detection.feature))
        self._next_id += 1

