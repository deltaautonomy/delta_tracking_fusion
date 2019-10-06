# Tracking and Fusion Pipeline

This package has one ROS node for for tracking and fusion.

### Topics

Following are the input topics to this node.
- `/delta/perception/ipm/camera_track` of type `delta_perception.CameraTrackArray`.
- `/carla/ego_vehicle/radar/tracks` of type `radar_msgs.RadarTrackArray`.
- `/delta/prediction/ego_vehicle/state` of type `delta_prediction.EgoStateEstimate`.

Following are the output topics from this node.
- `/delta/tracking_fusion/tracker/tracks` of type `delta_tracking_fusion.TrackArray`.
- `/delta/tracking_fusion/tracker/occupancy_grid` of type `delta_tracking_fusion.OccupancyGrid`.
- `/delta/tracking_fusion/tracker/track_id_marker` of type  `delta_tracking_fusion.Marker`

### Usage

Run the following command to execute this node.
```
rosrun delta_tracking_fusion main.py
```
