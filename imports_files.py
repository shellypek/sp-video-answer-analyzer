import torchaudio
import whisper
from openai import OpenAI
import torch
import fastapi
import pydantic
import requests
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import os
import uvicorn
import re
import csv
import shutil
import numpy as np
import sklearn
import random
import numbers
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
from torchvision import transforms
import functools
import librosa
from torch import nn
import torch.utils.data as data
from multimodal.models.multimodalcnn import MultiModalCNN
from interview_classes import EmotionResult
from multimodal.opts import opts
from multimodal.ravdess_preprocessing.process import extract_fa
from utils import calculate_accuracy
from multimodal.transforms import Compose, ToTensor
from torch.autograd import Variable        
import cv2
import soundfile as sf
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import librosa.display
from moviepy.editor import VideoFileClip
from ravdess_preprocessing.create_annotations import create_annotations
import torch.nn.functional as F
