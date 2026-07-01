# Deployment Guide

Platform

Render

Requirements

Python 3.10

Build

pip install -r requirements.txt

Start Command

uvicorn main:app --host 0.0.0.0 --port $PORT

Verify

GET /health

POST /chat

Public URL

To be added after deployment.