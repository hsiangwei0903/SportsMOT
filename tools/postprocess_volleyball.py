from logging import raiseExceptions
import os
import os.path as osp
import numpy as np
from scipy.spatial import distance
import glob
import collections

def check_sport(seq,sports_dic):
    for sport in sports_dic:
        if seq in sports_dic[sport]:
            return sport

class basetrack(object):
    def __init__(self,box,t_id):
        self.box = box
        self.id = t_id
        self.mapping = set()
        self.active = True
    def update(self,box):
        self.box = box
        self.active = True

sports = ['basketball','football','volleyball']
video_dic = {}
splits_dir = '/work/u1436961/hsiangwei/dataset/splits_txt'
for sport in sports: 
    file_path = osp.join(splits_dir,sport)
    my_file = open(file_path+".txt", "r")
    content = my_file.read()
    video_dic[sport] = content.split("\n")
    my_file.close()

seqs = os.listdir('/home/u1436961/hsiangwei/sportsmot/OC_SORT/result/final/')
for s_id,seq in enumerate(seqs):
    seq = seq.replace('.txt','')
    sport_name = check_sport(seq,video_dic)
    if sport_name != 'volleyball':
        continue
    print('processing ',seq,' ',s_id+1)
    txt = '/home/u1436961/hsiangwei/sportsmot/OC_SORT/result/final/{}.txt'.format(seq)
    path_out = '/home/u1436961/hsiangwei/sportsmot/OC_SORT/post_processing/{}.txt'.format(seq)
    labels = np.genfromtxt(txt, delimiter=',', dtype=None)
    labels = np.sort(labels,axis=0)
    d = collections.defaultdict()
    all_ids = set()
    online_ids = set()
    tracks = []
    label_idx = 0
    max_frame = labels[-1][0]+1
    for frame_id in range(1,max_frame):
        for idx in range(len(tracks)):
            tracks[idx].active = False
        for idx in range(len(tracks)):
            assert(tracks[idx].active == False)
        while label_idx<len(labels)-1 and labels[label_idx][0]==frame_id:
            label = labels[label_idx]
            label_idx+=1
            box = (label[2]+label[4]/2,label[3]+label[5]/2)
            if label[1] not in all_ids and label[1]<=12: # initiate tracks
                all_ids.add(label[1])
                trk = basetrack(box,label[1])
                tracks.append(trk)
            elif label[1] in all_ids and label[1]<=12: # same ID association
                for idx in range(len(tracks)):
                    if tracks[idx].id == label[1]:
                        tracks[idx].update(box)
                        break
            elif label[1] in all_ids and label[1]>12: # already handled re-enty mapping
                for idx in range(len(tracks)):
                    if label[1] in tracks[idx].mapping:
                        tracks[idx].update(box)
                        break
            elif label[1] not in all_ids and label[1]>12: # re-entry that requires handle
                min_idx = None
                min_dist = None
                for idx in range(len(tracks)):
                    if tracks[idx].active == False:
                        if min_dist is None:
                            min_dist = distance.euclidean(box,tracks[idx].box)
                            min_idx = idx
                        elif min_dist > distance.euclidean(box,tracks[idx].box):
                            min_dist = distance.euclidean(box,tracks[idx].box)
                            min_idx = idx
                if min_dist is None or min_dist > 400:
                    print(label[1])
                    print('not associated')
                    trk = basetrack(box,label[1])
                    tracks.append(trk)
                    all_ids.add(label[1])
                elif min_dist<400:
                    all_ids.add(label[1])
                    tracks[min_idx].update(box)
                    tracks[min_idx].mapping.add(label[1])
    
    mapper = {}
    for idx in range(len(tracks)):
        mapper[tracks[idx].id] = tracks[idx].mapping
        mapper[tracks[idx].id].add(tracks[idx].id)
    #print(mapper)
    file = open(path_out,"w")
    
    frame_to_id = collections.defaultdict(list)
    
    for label in labels:
        old_id = label[1]
        for m in mapper:
            if old_id in mapper[m]:
                new_id = m
                break
        if new_id not in frame_to_id[label[0]]:
            frame_to_id[label[0]].append(new_id)
            line = ','.join([str(label[0]),str(int(new_id)),str(label[2]),str(label[3]),str(label[4]),str(label[5]),str(label[6]),str(label[7]),str(label[8]),str(label[9])])
            file.write(line)
            file.write("\n")
        else:
            line = ','.join([str(label[0]),str((label[1]+100)),str(label[2]),str(label[3]),str(label[4]),str(label[5]),str(label[6]),str(label[7]),str(label[8]),str(label[9])])
            file.write(line)
            file.write("\n")