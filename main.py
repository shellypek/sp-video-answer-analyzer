from imports_files import *
from interview_classes import VideoRequest, EmotionResult, Question, Result, InterviewResults, QuestionReq, Request
from multimodal import prediction



client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get('API_KEY'),
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = whisper.load_model("base", device=device)

def Speech2Text(file):
    result = model.transcribe(file)
    answer = result["text"]
    return answer

def download_video(url, filename):
    logging.info(f"Downloading video from {url} to {filename}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for non-200 status codes

        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(1024):
                downloaded += len(chunk)
                f.write(chunk)
                if total_size > 0:
                    print(f"Downloaded {downloaded}/{total_size} bytes ({downloaded/total_size:.2%})")

        return True

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading video: {e}")
        return False

def ChatGPTEval(question, answer):
    message = [
        {
            "role": "user",
            "content": f"""Here's the question: "{question}" and here's the answer: "{answer}". Please provide your assessment based on the following criteria:
            Answer score: 
            - Rate the candidate's answer as an integer from 0 to 100, where 0 is the worst and 100 is the best.

            Score explanation:
            - Choose the score category based on the given score: 
            91-100: Excellent (The answer is exceptional, showing deep understanding and creativity.) 
            71-90: Good (The answer demonstrates solid knowledge and good application, though minor gaps may be present.)
            51-70: Satisfactory (The answer covers basic concepts correctly but lacks depth or detail.) 
            31-50: Poor (The answer shows some understanding but is largely incorrect or irrelevant.) 
            0-30: Very Poor (The answer is mostly incorrect, irrelevant, or not provided.)
            
            Question type:
            - Specify if the question is Technical, Behavioral, or Motivational. 
            
            Emotion:
            - Classify the candidate's emotional tone during the answer from the following emotions: 
            Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

            """
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=message,
        model="gpt-4",
    )

    content = chat_completion.choices[0].message.content

    # Define regular expressions for each field
    score_regex = r"Answer score:\s*(.*?)(?=\s*Score explanation)"
    explanation_regex = r"Score explanation:\s*(.*?)(?=\s*Question type)"
    type_regex = r"Question type:\s*(.*?)(?=\s*Emotion)"
    emotion_regex = r"Emotion:\s*(.*)"

    # Extract information using regex matches
    answer_score = re.search(score_regex, content).group(1).strip()
    score_explanation = re.search(explanation_regex, content).group(1).strip()
    question_type = re.search(type_regex, content).group(1).strip()
    emotion = re.search(emotion_regex, content).group(1).strip()

    return answer_score, score_explanation, question_type, emotion

app = FastAPI()

def ensure_video_folder_exists(path):
    """Creates the 'video' folder if it doesn't exist."""

    if not os.path.exists(path):
        os.makedirs(path)


@app.post("/process_interview")
async def process_interview(interview: Request):
    print(interview)
    
    interview_results = InterviewResults(result=Result(questions=[], score=0), rawResult=b"")
    result = Result(questions=[], score=0)
    
    for question in interview.questions:
        filename = f"interview_{question.public_id}"
        video_path = os.path.join("video" + filename, filename + "video.mp4")  # Combine folder and filename

        ensure_video_folder_exists("video" + filename)

        if download_video(question.video_link, video_path):
            try:
                answer = Speech2Text(video_path)
            except Exception as e:
                answer = f"Error processing video: {e}"
                os.remove(video_path)
        else:
            answer = "Error downloading video"
            os.remove(video_path)
        print(answer)
        ans = answer
        answer_score, score_explanation, question_type, emotion = ChatGPTEval(question.question, answer)
        emotion_results = prediction.PredictionVideo("video" + filename)  # Use video_path instead of video_link

        question_obj = Question(
            question=question.question,
            score=int(answer_score),
            video_link=question.video_link,
            question_type=question_type,
            evaluation=score_explanation,
            emotion=emotion,
            answer=ans,
            public_id=question.public_id,
            emotion_results = emotion_results
        )
        print("he",ans)
        result.questions.append(question_obj)
        result.score += int(answer_score)
        print(question_obj)
        os.remove(video_path)
    interview_results.result = result
    print(interview_results)
    return {"result": interview_results.result}


