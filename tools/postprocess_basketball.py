from logging import raiseExceptions
import os
import os.path as osp
import numpy as np
from scipy.spatial import distance
import glob
import collections

# check sport category
def check_sport(seq,sports_dic):
    for sport in sports_dic:
        if seq in sports_dic[sport]:
            return sport
# base tracklet class
class basetrack(object):
    def __init__(self,box,t_id,feature):
        self.box = box
        self.id = t_id
        self.feature = feature
        self.alpha = 0.9
        self.mapping = set()
        self.active = True
    def update(self,box,feature):
        self.box = box
        # update feature using exponential moving average
        self.feature = self.alpha*self.feature+(1-self.alpha)*feature
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
    if sport_name != 'basketball':
        continue
    print('processing ',seq,' ',s_id+1)
    txt = '/home/u1436961/hsiangwei/sportsmot/OC_SORT/result/final/{}.txt'.format(seq)
    embs = np.load('/home/u1436961/hsiangwei/sportsmot/embedding/emb2/{}.npy'.format(seq),allow_pickle = True)
    path_out = '/home/u1436961/hsiangwei/sportsmot/OC_SORT/post_processing/{}.txt'.format(seq)
    labels = np.genfromtxt(txt, delimiter=',', dtype=None)
    labels = np.sort(labels,axis=0)
    d = collections.defaultdict()
    all_ids = set()
    online_ids = set()
    tracks = []
    assert len(embs) == len(labels)
    label_idx = 0
    max_frame = labels[-1][0]+1
    for frame_id in range(1,max_frame):
        for idx in range(len(tracks)):
            tracks[idx].active = False
        for idx in range(len(tracks)):
            assert(tracks[idx].active == False)
        while label_idx<len(labels)-1 and labels[label_idx][0]==frame_id:
            label = labels[label_idx]
            emb = embs[label_idx]
            label_idx+=1
            box = (label[2]+label[4]/2,label[3]+label[5]/2)
            if label[1] not in all_ids and label[1]<=10: # initiate tracks if id less then 10
                all_ids.add(label[1])
                trk = basetrack(box,label[1],emb)
                tracks.append(trk)
            elif label[1] in all_ids and label[1]<=10: # same ID association, update features
                for idx in range(len(tracks)):
                    if tracks[idx].id == label[1]:
                        tracks[idx].update(box,emb)
                        break
            elif label[1] in all_ids and label[1]>10: # already handled re-enty mapping, update features
                for idx in range(len(tracks)):
                    if label[1] in tracks[idx].mapping:
                        tracks[idx].update(box,emb)
                        break
            elif label[1] not in all_ids and label[1]>10: # re-entry that requires handle
                min_idx = None
                min_dist = None
                for idx in range(len(tracks)):
                    if tracks[idx].active == False:
                        if min_dist is None:
                            min_dist = distance.cosine(emb,tracks[idx].feature)
                            min_idx = idx
                        elif min_dist > distance.cosine(emb,tracks[idx].feature):
                            min_dist = distance.cosine(emb,tracks[idx].feature)
                            min_idx = idx
                if min_dist is None: # exception can occur, we stack up the global ID (it can be more than 10 in basketball)
                    # print(label[1])
                    # print('not associated')
                    trk = basetrack(box,label[1],emb)
                    tracks.append(trk)
                    all_ids.add(label[1])
                else:
                    all_ids.add(label[1])
                    tracks[min_idx].update(box,emb)
                    tracks[min_idx].mapping.add(label[1])
    
    mapper = {}
    for idx in range(len(tracks)):
        mapper[tracks[idx].id] = tracks[idx].mapping
        mapper[tracks[idx].id].add(tracks[idx].id)
    #print(mapper)
    file = open(path_out,"w")
    
    frame_to_id = collections.defaultdict(list)
    orig_set = set([num for num in range(1,11)])
    
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