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
            "content": f"""Rate the candidate's answer as an integer from 0 to 100, where 0 is the worst and 100 is the best and give a little bit explanation. Also, provide type of the question: technical, behavioral, or motivation. 
            could you please give emotions class for the answer from the following emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised.
            Here's the question: "{question}" and here's the answer: "{answer}".
            Please give your answer in the following format:
            Answer score:
            Score explanation:
            Question type:
            Emotion:
            """
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=message,
        model="gpt-3.5-turbo",
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

def ensure_video_folder_exists():
    """Creates the 'video' folder if it doesn't exist."""
    video_folder_path = "video"
    if not os.path.exists(video_folder_path):
        os.makedirs(video_folder_path)


@app.post("/process_interview")
async def process_interview(interview: Request):
    print(interview)
    
    interview_results = InterviewResults(result=Result(questions=[], score=0), rawResult=b"")
    result = Result(questions=[], score=0)
    
    for question in interview.questions:
        filename = f"interview_{question.public_id}.mp4"
        video_path = os.path.join("video", filename)  # Combine folder and filename

        ensure_video_folder_exists()

        if download_video(question.video_link, video_path):
            try:
                answer = Speech2Text(video_path)
                os.remove(video_path)
            except Exception as e:
                answer = f"Error processing video: {e}"
        else:
            answer = "Error downloading video"

        ans = answer
        answer_score, score_explanation, question_type, emotion = ChatGPTEval(question.question, answer)
        emotion_results = prediction(video_path)  # Use video_path instead of video_link

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
    interview_results.result = result
    print(interview_results)
    return {"result": interview_results.result}

