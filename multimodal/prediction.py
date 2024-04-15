from imports_files import *
import time
def video_loader(video_dir_path):
    video = np.load(video_dir_path)    
    video_data = []
    for i in range(np.shape(video)[0]):
        video_data.append(Image.fromarray(video[i,:,:,:]))    
    return video_data

def get_default_video_loader():
    return functools.partial(video_loader)

def load_audio(audiofile, sr):
    audios = librosa.core.load(audiofile, sr=sr)
    y = audios[0]
    return y, sr

def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return mfcc

def make_dataset(subset, annotation_path):
    with open(annotation_path, 'r') as f:
        annots = f.readlines()
        
    dataset = []
    for line in annots:
        filename, audiofilename, label, trainvaltest = line.split(';')        
        if trainvaltest.rstrip() != subset:
            continue
        
        sample = {'video_path': filename,                       
                  'audio_path': audiofilename, 
                  'label': int(label)-1}
        dataset.append(sample)
    return dataset 
       

class RAVDESS(data.Dataset):
    def __init__(self,                 
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 get_loader=get_default_video_loader, data_type = 'audiovisual', audio_transform=None):
        self.data = make_dataset(subset, annotation_path)
        self.spatial_transform = spatial_transform
        self.audio_transform=audio_transform
        self.loader = get_loader()
        self.data_type = data_type 

    def __getitem__(self, index):
        target = self.data[index]['label']
                

        if self.data_type == 'video' or self.data_type == 'audiovisual':        
            path = self.data[index]['video_path']
            clip = self.loader(path)
            
            if self.spatial_transform is not None:               
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]            
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3) 
            
            if self.data_type == 'video':
                return clip, target
            
        if self.data_type == 'audio' or self.data_type == 'audiovisual':
            path = self.data[index]['audio_path']
            y, sr = load_audio(path, sr=22050) 
            
            if self.audio_transform is not None:
                 self.audio_transform.randomize_parameters()
                 y = self.audio_transform(y)     
                 
            mfcc = get_mfccs(y, sr)            
            audio_features = mfcc 

            if self.data_type == 'audio':
                return audio_features, target
        if self.data_type == 'audiovisual':
            return audio_features, clip, target  
        
    def __len__(self):
        return len(self.data)

def get_test_set(opt, spatial_transform=None, audio_transform=None):

    subset = 'testing'
    
    test_data = RAVDESS(
        opt.annotation_path,
        subset,
        spatial_transform=spatial_transform, data_type='audiovisual',audio_transform=audio_transform)
    return test_data

def PredictionVideo(root):
    if extract_fa(root) == False:
        return
    fps = 30    
    for filename in os.listdir(root):
        if filename.endswith('.mp4'):
            cap = cv2.VideoCapture(os.path.join(root,filename))  
            fps = cap.get(cv2.CAP_PROP_FPS)
            
    opt = opts

    video_transform = Compose([
                    ToTensor(255)])
        
    test_data = get_test_set(opt, spatial_transform=video_transform) 

    model = MultiModalCNN(opt.n_classes, fusion = opt.fusion, seq_length = opt.sample_duration, pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)
    model = model.to(opt.device)
    model = nn.DataParallel(model, device_ids=None)
    #model.load_state_dict(torch.load('d:/RAVDESS/RAVDESS_multimodalcnn_15_best0.pth'))
    best_state = torch.load('/app/multimodal/RAVDESS_multimodalcnn_15_best0.pth', map_location=torch.device(opt.device))
    model.load_state_dict(best_state['state_dict'])
    model.eval()


    predictions = []

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ans=np.array([])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, (inputs_audio, inputs_visual, targets) in enumerate(test_loader):
        inputs_visual = inputs_visual.permute(0, 2, 1, 3, 4)
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0] * inputs_visual.shape[1], inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4])

        targets = targets.to(opt.device)
        with torch.no_grad():
            inputs_visual = inputs_visual.to(device)
            inputs_audio = inputs_audio.to(device)
            targets = targets.to(device)
            outputs = model(inputs_audio, inputs_visual)
            #print(outputs)
            _, preds = torch.max(outputs, 1)

            # print(preds)

            ans=np.concatenate((ans,preds.cpu().numpy())) 



    ans_list=ans.tolist()
            
    switcher={
        0: "Neutral",
        1: "Calm", 
        2: "Happy",
        3: "Sad", 
        4: "Angry", 
        5: "Fearful", 
        6: "Disgust", 
        7: "Surprised"
    }

    # timepoints=[]
    # start=0
    # for i in range(len(ans_list)):
    #     if i!=0 and ans_list[i-1]!=ans_list[i]:
    #         timepoints.append({"emotion":switcher.get(ans_list[i-1]),"start":start, "duration":round(float((i*35+1)/30)-start)})
    #         start=round(float((i*35+1)/30))
    # timepoints.append({"emotion":switcher.get(ans_list[-1]),"start":start,"duration": round(float(((len(ans_list)-1)*35+1)/30)-start)})

    # print(timepoints)
    # return timepoints
    timepoints = []
    if len(ans_list) != 0:
        start = 0
        i = 0
        for i in range(len(ans_list)):
            if i != 0 and ans_list[i-1] != ans_list[i]:
                timepoints.append(EmotionResult(emotion=switcher.get(ans_list[i-1]), exact_time=start, duration=round(float((i*35+1)/fps)-start, 1)))
                start = round(float((i*35+1)/fps), 1)
        print(ans_list)
        print(i)
        timepoints.append(EmotionResult(emotion=switcher.get(ans_list[i]), exact_time=start, duration=round(float(((len(ans_list))*35+1)/fps)-start, 1)))
    

    return timepoints
    




# prediction("C:/Users/zhk27/OneDrive/Рабочий стол/SP/myproject/media/videos")

#prediction("d:/django senior project/SP/myproject/media/videos")
