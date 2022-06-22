import csv
import os
from operator import itemgetter
from pathlin import Path

filename='epic-kitchens-100-annotations/EPIC_100_train.csv'
frames_path = '/vast/irr2020/EPIC-KITCHENS/frames'
ann_lines = []
with open(filename, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        ann_lines.append(row)

ann_lines = ann_lines[1:]

# sort by participants
sorted(ann_lines, key=itemgetter(1))

# split participants
PNN_dict={}
for line in ann_lines:
    video_id = line[2] 
    participant_id = line[1] 
    narration = line[8]
    start_frame = int(line[6])
    stop_frame = int(line[7])
    if participant_id not in PNN_dict: PNN_dict[participant_id] = {}
    if video_id not in PNN_dict[participant_id]: PNN_dict[participant_id][video_id] = []
    PNN_dict[participant_id][video_id].append([narration, start_frame, stop_frame])

for participant_id, PNN_NN_dict in PNN_dict.items():
    for video_id, act_stamps in PNN_NN_dict.items():
        # see if video frames exist
        try:
            if not os.path.isdir(os.path.join(frames_path,participant_id,video_id)):
                raise ValueError('participant {} video {} not found'.format(participant_id,video_id))
        except ValueError:
            continue

        # get labels path and create it if needed
        labels_path = os.path.join(frames_path[:6],'labels',participant_id,video_id)
        try:
            os.mkdir(labels_path)
        except OSError:
            pass

        for act_stamp in act_stamps:
            action, start_frame, stop_frame = act_stamp
            start_frame = start_frame + 1 if start_frame % 2 == 1 else start_frame

            for frame in range(start_frame,stop_frame,2):
                assert os.path.isfile(os.path.join(frames_path, '{:07}.jpg'.format(frame)))
                frame_file = os.path.join(labels_path,'{:07d}.txt'.format(frame))
                try:
                    Path(frame_file).touch()
                except OSError:
                    pass
                with open(frame_file, 'a') as f:
                    f.write(action + '\n')