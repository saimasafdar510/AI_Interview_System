# AI Interview System

An AI and Machine Learning Based Voice Interview Practice System built with Python and Django.

This project provides a web-based environment where candidates can practice technical interviews, answer questions, and receive structured feedback based on their responses.

## Project Overview

The AI Interview System aims to make technical interview preparation easier through an interactive interview simulation platform.

Users can practice technical questions, complete interview sessions, and review their assessment results.

## Features

- Django-based web application
- Technical interview question bank
- Skill and difficulty-based question selection
- Interview session management
- Answer submission and evaluation
- Structured feedback and final assessment
- Experimental Machine Learning answer classifier
- Browser-based speech recognition and synthesis components
- Experimental speaker verification

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Django | Web framework |
| SQLite | Development database |
| Scikit-learn | Experimental ML classifier |
| TF-IDF | Text feature extraction |
| Logistic Regression | Answer classification |
| SpeechBrain | Experimental speaker verification |
| HTML, CSS | Frontend |
| JavaScript | Browser interaction |
| JSON | Question bank |
| uv | Dependency management |

## Machine Learning Component

The project includes an experimental answer classifier using:

- TF-IDF Vectorization
- Word n-grams
- Logistic Regression
- Scikit-learn

It classifies answers into three categories:

- Poor
- Average
- Good

**Important:** The ML classifier is experimental and separate from the main rule-based evaluation engine. Its current performance is limited and should not be considered a reliable measure of candidate ability.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/saimasafdar510/AI_Interview_System.git
```

### 2. Open project folder

```bash
cd AI_Interview_System
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Apply migrations

```bash
uv run python manage.py migrate
```

### 5. Run the server

```bash
uv run python manage.py runserver
```

Open your browser:

```text
http://127.0.0.1:8000/
```

## Project Status

This project is currently under development. Machine Learning classification and voice-related features are experimental and require further testing and improvement.

## Future Improvements

- Improve ML model accuracy with a larger labelled dataset.
- Enhance interview feedback and evaluation.
- Improve voice interaction and speech processing.
- Expand technical question categories.
- Improve user interface and experience.
- Add detailed performance analytics.

## Developer

**Saima Safdar**

GitHub: [@saimasafdar510](https://github.com/saimasafdar510)

---

Developed as an academic Final Year Project (FYP).