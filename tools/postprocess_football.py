import os
import os.path as osp
import numpy as np
from scipy.spatial import distance
import collections

# check sport category
def check_sport(seq,sports_dic):
    for sport in sports_dic:
        if seq in sports_dic[sport]:
            return sport
# base tracklet class
def get_dist(tracklet_a,tracklet_b):
    length = min(len(tracklet_a.feature),len(tracklet_b.feature))
    dist = 0
    for idx in range(length):
        dist += distance.cosine(tracklet_a.feature[idx],tracklet_b.feature[idx])
    return dist/length

def distance_thres(time_gap):
    assert time_gap>0
    if time_gap<100:
        return 100
    elif 100<=time_gap<500:
        return 250
    elif 500<=time_gap:
        return 400

class basetrack(object):
    def __init__(self,box,t_id,feature,frame):
        self.box_start = box
        self.box_end = None
        self.id = t_id
        self.feature = [feature]
        self.active = False
        self.start = frame
        self.end = None
    def update(self,box,feature,frame):
        self.box_end = box
        self.feature.append(feature)
        self.end = frame

if __name__ == "__main__":
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
    cnt = 0
    for s_id,seq in enumerate(seqs):
        seq = seq.replace('.txt','')
        sport_name = check_sport(seq,video_dic)
        if sport_name != 'football':
            continue
        print('processing ',seq,' ',s_id+1)
        cnt += 1
        txt = '/home/u1436961/hsiangwei/sportsmot/OC_SORT/result/final/{}.txt'.format(seq)
        embs = np.load('/home/u1436961/hsiangwei/sportsmot/embedding/emb3/{}.npy'.format(seq),allow_pickle = True)
        labels = np.genfromtxt(txt, delimiter=',', dtype=None)
        labels = np.sort(labels,axis=0)
        tracks = collections.defaultdict()
        assert len(embs) == len(labels)
        for label_idx,label in enumerate(labels):
            frame_id = label[0]
            t_id = label[1]
            box = (label[2]+label[4]/2,label[3]+label[5]/2)
            feature = embs[label_idx]
            if t_id not in tracks:
                tracks[t_id] = basetrack(box,t_id,feature,frame_id)
            else:
                tracks[t_id].update(box,feature,frame_id)
                
        t_lists = []
        for t_id in tracks:
            if tracks[t_id].end is not None:
                t_lists.append(tracks[t_id])
        
        mapper = collections.defaultdict(list)
        
        # first round association
        for starter in range(len(t_lists)):
            if t_lists[starter].active: # tracklet already associated
                continue
            associated = False
            end = t_lists[starter].end
            for associater in range(starter+1,len(t_lists)):
                if not t_lists[associater].active:
                    box_distance = distance.euclidean(t_lists[starter].box_end,t_lists[associater].box_start)
                    if t_lists[associater].start > end and box_distance < distance_thres(t_lists[associater].start - end):
                        emb_distance = get_dist(t_lists[starter],t_lists[associater])
                        if emb_distance < 0.1:
                            end = t_lists[associater].end # change ending point
                            mapper[t_lists[starter].id].append(t_lists[associater].id) # update mapper
                            t_lists[associater].active = True # prevent double association
                            t_lists[starter].box_end = t_lists[associater].box_end # position update
                            t_lists[starter].feature = t_lists[associater].feature # feature update
                            associated = True
            if associated:
                t_lists[starter].active = True
                mapper[t_lists[starter].id].append(t_lists[starter].id)
                
        # second round association
        for starter in range(len(t_lists)):
            if t_lists[starter].active: # tracklet already associated
                continue
            associated = False
            end = t_lists[starter].end
            for associater in range(starter+1,len(t_lists)):
                if not t_lists[associater].active:
                    box_distance = distance.euclidean(t_lists[starter].box_end,t_lists[associater].box_start)
                    if t_lists[associater].start > end and box_distance < distance_thres(t_lists[associater].start - end):
                        emb_distance = get_dist(t_lists[starter],t_lists[associater])
                        if emb_distance < 0.2:
                            end = t_lists[associater].end # change ending point
                            mapper[t_lists[starter].id].append(t_lists[associater].id) # update mapper
                            t_lists[associater].active = True # prevent double association
                            t_lists[starter].box_end = t_lists[associater].box_end # position update
                            t_lists[starter].feature = t_lists[associater].feature # feature update
                            associated = True
            if associated:
                t_lists[starter].active = True
                mapper[t_lists[starter].id].append(t_lists[starter].id)
        
        # third round association                    
        for starter in range(len(t_lists)):
            if t_lists[starter].active: # tracklet already associated
                continue
            t_lists[starter].active = True
            mapper[t_lists[starter].id].append(t_lists[starter].id)
            end = t_lists[starter].end
            for associater in range(starter+1,len(t_lists)):
                if not t_lists[associater].active:
                    box_distance = distance.euclidean(t_lists[starter].box_end,t_lists[associater].box_start)
                    if t_lists[associater].start > end and box_distance < distance_thres(t_lists[associater].start - end):
                        emb_distance = get_dist(t_lists[starter],t_lists[associater])
                        if emb_distance < 0.4:
                            end = t_lists[associater].end # change ending point
                            mapper[t_lists[starter].id].append(t_lists[associater].id) # update mapper
                            t_lists[associater].active = True # prevent double association
                            t_lists[starter].box_end = t_lists[associater].box_end # position update
                            t_lists[starter].feature = t_lists[associater].feature # feature update
        
        #print(mapper)   
        path_out = '/home/u1436961/hsiangwei/sportsmot/OC_SORT/post_processing/{}.txt'.format(seq)
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