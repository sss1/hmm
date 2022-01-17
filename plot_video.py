import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle

# This script generates all of the videos that were used by human coders to
# hand-code the eye-tracking data.
# By changing prediction_to_plot below, the same script can be used to plot
# the predictions of the HMM and/or Naive model.

target_color = 'b' # Usually 'b'; None means random, for human coding
target_name = 'Target' # Usually 'Target'
distractor_color = 'r' # Usually 'r'

save_video = False # If True, saves the output video. If False, only displays the video.
root = './' # root directory 

# boundaries of track-it grid
x_min = 400
x_max = 2000
y_min = 0
y_max = 1200

space = 50 # number of extra pixels to display on either side of the plot

prediction_to_plot = 'HMM' # Should be one of 'HMM', 'Naive', or 'None'

lag = 10 # plot a time window of length lag, so we can see the trajectory more clearly

def plot_trial_video(observations: np.array, true_means: np.array, posterior_mode: np.array):

  # Set up formatting for the saved movie file
  if save_video:
    Writer = animation.writers['ffmpeg']
    relative_speed = 0.1
    original_fps = 60
  eyetrack = observations
  target = true_means[:, 0, :]
  distractors = true_means[:, 1:, :].transpose((1, 0, 2))
  if prediction_to_plot == 'HMM':
    MLE = posterior_mode
  elif prediction_to_plot == 'Naive':
    MLE = naive_eyetracking.get_trackit_MLE(np.swapaxes(eyetrack,0,1), np.swapaxes(target,0,1), np.swapaxes(distractors,1,2))
  
  trial_length = target.shape[0]
  
  # initializate plot background and objects to plot
  fig = plt.figure()
  ax = plt.axes(xlim=(x_min, x_max), ylim = (y_min, y_max))
  eyetrack_line, = ax.plot([], [], c = 'k', lw = 2, label = 'Eye-track')
  distractors_lines = []
  if prediction_to_plot != 'None':
    state_point = ax.scatter([], [], s = 75, c = 'g', label = 'Model Prediction')
  bg_color = plt.fill_between([x_min, x_max], [y_min, y_min], [y_max, y_max], alpha=0.0, color='grey')
  frame_text = ax.text(1800, 25, str(0))
  trackit_line, = ax.plot([], [], lw = 2, c = target_color, label = target_name)
  for j in range(len(distractors)):
    distractors_lines.extend(ax.plot([], [], c = distractor_color, lw = 2, label = 'Distractor ' + str(j + 1)))
  
  legend_entries = [eyetrack_line, trackit_line]
  legend_entries.extend(distractors_lines)
  if prediction_to_plot != 'None':
    legend_entries.append(state_point)
  plt.legend(handles = legend_entries, loc = 'upper right')
  
  # Rather than a single point, show tail of object trajectories (frames in range(trial_length - lag))
  # This makes it much easier to follow objects visually
  def animate(i):
    if i % 100 == 0:
      print('Current frame: ' + str(i))
    frame_text.set_text(str(i))
    trackit_line.set_data(target[i:(i + lag),0], target[i:(i + lag),1])
    eyetrack_line.set_data(eyetrack[i:(i + lag),0], eyetrack[i:(i + lag),1])
    for j in range(len(distractors)):
      distractors_lines[j].set_data(distractors[j,i:(i + lag),0],
                                    distractors[j,i:(i + lag),1])
    if prediction_to_plot != 'None':
      state = MLE[i + lag]
      if int(state) in {0, 7}:  # On Target, either in Distractible or On-Task
        state_point.set_offsets(target[i + lag - 1,:])
      elif state == 8 or state < 0:  # Off-Task or missing
        state_point.set_offsets([0, 0])
      else:  # On Distractor
        state_point.set_offsets(distractors[state - 1, i + lag - 1, :])

    off_task = (state == 8)
    missing_data = np.any(np.isnan(eyetrack[i:(i + lag),0]))

    # Indicate missing data with a red background, off-task with a green
    # background, and both with a grey background
    if off_task or missing_data:
      bg_color.set_alpha(0.2)
      if not off_task:
        bg_color.set_color('red')
      elif not missing_data:
        bg_color.set_color('green')
      else:
        bg_color.set_color('grey')
    else:
      bg_color.set_alpha(0.0)

    plt.draw()
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    return trackit_line, eyetrack_line,
  
  
  anim = animation.FuncAnimation(fig, animate,
                                 frames = trial_length - lag,
                                 interval = 8.33,
                                 blit = False,
                                 repeat = False)
  if save_video:
    video_dir = root
    save_path = video_dir + subject_ID + '_' + experiment + '_trial_' + str(trial_idx) + '_uncoded.mp4'
    print('Saving video to ' + save_path)
    anim.save(save_path, writer = writer)
  else:
    plt.show()
