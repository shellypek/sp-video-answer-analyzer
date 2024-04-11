from imports_files import *

class VideoRequest(BaseModel):
    questionNumber: int
    videoFile: bytes

class EmotionResult(BaseModel):
    emotion: str
    exact_time: float
    duration: float

class Question(BaseModel):
    public_id: str
    question: str
    evaluation: str
    score: int
    video_link: str
    question_type: str
    emotion: str
    emotion_results: list[EmotionResult]
    answer: str

class Result(BaseModel):
    questions: list[Question]
    score: int

class InterviewResults(BaseModel):
    result: Result
    rawResult: bytes

class QuestionReq(BaseModel):
    question: str
    public_id: str
    video_link: str

class Request(BaseModel):
    questions: list[QuestionReq]