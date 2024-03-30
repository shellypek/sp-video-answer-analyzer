from imports_files import *

client = OpenAI(
    # This is the default and can be omitted
    api_key= os.environ.get('API_Key'),
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
        "content": f"""Rate the candidate's answer as an integer from 0 to 100, where 0 is the worst and 100 is the best and give a little bit explanation. Also, provide type of the question: technical, behavioural, or motivation. 
        If it's behavioral and motivational question, could you please give emotions class for the answer from the following emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised.
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

class VideoRequest(BaseModel):
    questionNumber: int
    videoFile: bytes

class EmotionResult(BaseModel):
    emotion: str
    exact_time: float
    duration: float

class Question(BaseModel):
    question: str
    evaluation: str
    score: int
    video_link: str
    question_type: str
    emotion_results: list[EmotionResult]

class Result(BaseModel):
    questions: list[Question]
    score: int

class InterviewResults(BaseModel):
    public_id: str
    result: Result
    rawResult: bytes

@app.post("/process_interview")
def process_interview(interview: InterviewResults, download_service, speech_service):
    public_id = interview.public_id
    result = interview.result

    for question in result.questions:
        filename = f"interview_{public_id}_{question.questionNumber}.mp4"
        if download_video(question.video_link, filename, download_service):
            try:
                answer = Speech2Text(filename, speech_service)
                os.remove(filename) 
            except Exception as e:
                answer = f"Error processing video: {e}"
        else:
            answer = "Error downloading video"

        answer_score, score_explanation, question_type, emotion = ChatGPTEval(question.question, answer)
        question.evaluation = score_explanation
        question.score = int(answer_score)
        question.question_type = question_type
        question.emotion = emotion

    return {"result": result}
