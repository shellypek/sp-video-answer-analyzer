# -*- coding: utf-8 -*-

import os
# root = 'c:/Users/zhk27/OneDrive/Рабочий стол/em_video'

def Create_annotations(root):
    #annotation_file = 'annotations_croppad_fold'+str(fold+1)+'.txt'
    annotation_file = 'annotations.txt'

    with open(annotation_file, 'w') as f:
        pass

    for video in os.listdir(root):
        if not video.endswith('.npy') or 'croppad' not in video:
            continue
        
        audio = video.split('_facecroppad')[0] + '_croppad.wav'  
        # print(str(audio))
        with open(annotation_file, 'a') as f:
            f.write(os.path.join(root, video) + ';' + os.path.join(root, audio) + ';'+ str(1) + ';testing' + '\n')





