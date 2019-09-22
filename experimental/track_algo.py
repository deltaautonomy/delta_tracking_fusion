'''

Pass all the active tracks through predict function of filter - get predicted states at t+1 ---- 1 kf.predict_step(current_time)
get ego vehicle state at timestep = t+1 ---------- 2
Do motion compensation in 1 according to 2 --------- 3
Get states measurement at timestep = t+1 from DA function ------- 4

get ego vehicle state at time step = t

Do DA between 3 and 4
re-assign tracks to matched detection-tracks and reset miss-countdown 
assign new tracks to detections that are not matched with the count down of 3 misses 
self.update_step(z_camera, z_radar, R_camera=None, R_radar=None)
for all active tracks - miss countdown  += 1

'''