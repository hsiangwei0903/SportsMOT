from numpy import genfromtxt
import glob 
import copy
import cv2
import os

limit = 2000
reid = True

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

if reid:
    path_in = glob.glob('C:/Users/USER/Desktop/aicity/txt/reid/*')
else:
    path_in = glob.glob('C:/Users/USER/Desktop/aicity/txt/no_reid/*')

viz_list = []

for path in path_in:
    viz_list.append(os.path.basename(path))

out_path = 'C:/Users/USER/Desktop/aicity/video'

for id,seq in enumerate(viz_list):
    seq = seq.replace('.txt','')
    scene,cam = seq.split('_')

    if cam!='c014':
        continue

    print('Plotting scene : {}  cam : {}'.format(scene,cam))
    labels = genfromtxt(path_in[id], delimiter=',', dtype=None)
    scene,camera = seq.split('_')
    imgs = sorted(glob.glob(os.path.join('C:/Users/USER/Desktop/aicity/test/',scene,camera,'img/*')))
    i = 0
    os.makedirs('C:/Users/USER/Desktop/aicity/result/{}'.format(seq),exist_ok=True)
    for frame,img in enumerate(imgs):
        if frame == limit:
            break
        frame += 1
        if frame%100 == 0:
            print('processing frame: ',frame)
        im = cv2.imread(img)
        im_out = copy.deepcopy(im)

        while labels[i][0] == frame:
            cv2.rectangle(im_out,(int(labels[i][2]),int(labels[i][3])),(int(labels[i][2])+int(labels[i][4]),int(labels[i][3])+int(labels[i][5])),get_color(int(labels[i][1])+1),2)
            cv2.putText(im_out,str(int(labels[i][1])),(int(labels[i][2]),int(labels[i][3])),cv2.FONT_HERSHEY_PLAIN, 
                                                                        max(1.0, im.shape[1]/1200),(0,255,255),thickness = 2)

            i += 1
            if i == len(labels):
                break

        cv2.imwrite('C:/Users/USER/Desktop/aicity/result/{}/{}.jpg'.format(seq,"%06d"%frame),im_out)
    
    path_inn = 'C:/Users/USER/Desktop/aicity/result/'
    if reid:
        output = os.path.join(out_path,seq+'_reid.mp4')
    else:
        output = os.path.join(out_path,seq+'_no_reid.mp4')
    cmd_str = 'ffmpeg -f image2 -i {} -c:v copy -vcodec libx264 -crf 25 {}'.format(path_inn+seq+'/'+'%06d'+'.jpg',output)
    os.system(cmd_str)