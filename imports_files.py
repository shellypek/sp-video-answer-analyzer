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
