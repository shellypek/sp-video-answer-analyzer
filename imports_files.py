import torchaudio
import whisper
import openai
import torch
import fastapi
import pydantic
import requests
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import os
