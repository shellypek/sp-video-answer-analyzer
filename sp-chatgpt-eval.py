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

import requests

def download_video(url, filename):
  response = requests.get(url, stream=True)
  if response.status_code == 200:
    with open(filename, 'wb') as f:
      for chunk in response.iter_content(1024):
        f.write(chunk)
    return True
  else:
    print(f"Error downloading video: {response.status_code}")
    return False

def ChatGPTEval(question, answer):
    message = [
    {
        "role": "user",
        "content": f"""Rate the candidate's answer as an integer from 0 to 100, where 0 is the worst and 100 is the best and give a little bit explanation. Also, provide type of the question: technical, behavioural, or motivation. 
        Here's the question: "{question}" and here's the answer: "{answer}".  
        Please give your answer in the following format: 
        Answer score:
        Score explanation:
        Question type: 
        """
    }
    ]

    chat_completion = client.chat.completions.create(
    messages=message,
    model="gpt-3.5-turbo",
    )

    # Assuming you have chat_completion.choices[0].message.content available
    content = chat_completion.choices[0].message.content

    # Extracting answer score
    answer_score_index = content.find("Answer score:") + len("Answer score:")
    score_explanation_index = content.find("Score explanation:")
    question_type_index = content.find("Question type:")

    answer_score = content[answer_score_index:score_explanation_index].strip()
    score_explanation = content[score_explanation_index:question_type_index].strip()
    question_type = content[question_type_index:].strip()

    return answer_score, score_explanation, question_type

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
def process_interview(interview: InterviewResults):
  public_id = interview.public_id
  result = interview.result

  for question in result.questions:
    filename = f"interview_{public_id}_{question.questionNumber}.mp4"  # Example filename
    if download_video(question.video_link, filename):
      answer = Speech2Text(filename)
      os.remove(filename)  # Clean up downloaded video
    else:
      answer = "Error downloading video"
    answer_score, score_explanation, question_type = ChatGPTEval(question.question, answer)
    question.evaluation = score_explanation
    question.score = int(answer_score)
    question.question_type = question_type

  return {"result": result}