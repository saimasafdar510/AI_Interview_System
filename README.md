# Machine Learning Based Voice Interview Simulator

A web-based interview practice application developed using Python and Django. The system helps candidates practice technical interviews, submit answers, and receive structured feedback through an evaluation system.

The project also includes an experimental Machine Learning classifier for answer classification and components for voice interaction and speaker verification.

> **Project Status:** Under active development. The main answer evaluation uses rule-based scoring. The Machine Learning classifier is experimental and does not control or replace the main evaluation score.

## Project Overview

The Machine Learning Based Voice Interview Simulator provides a structured environment where users can:

- Practice technical interview questions.
- Select interview skills and difficulty levels.
- Attempt interview sessions.
- Submit answers for evaluation.
- Receive structured assessment results.
- Review their interview performance.

The project is designed to explore how web development, rule-based evaluation, Machine Learning, and voice technologies can work together in an interview practice application.

## Main Features

### 1. Django Web Application
- Django-based backend.
- Interview session management.
- Application views and URL routing.
- Web-based user interface.

### 2. Question Bank
- Technical interview questions.
- Question categories and skills.
- Difficulty-based question selection.
- JSON and database-based question management.

### 3. Interview Management
- Interview session creation.
- Question and answer management.
- Interview progress tracking.
- Final assessment components.

### 4. Answer Evaluation
The main evaluation engine uses predefined rules and scoring criteria.

It evaluates answers using elements such as:
- Keywords.
- Expected concepts.
- Relevant answer content.
- Scoring categories.

The rule-based evaluation provides the primary assessment score.

### 5. Experimental Machine Learning Classifier

The project includes an experimental text classification model.

**Machine Learning technologies:**
- TF-IDF Vectorization.
- Word n-grams (1–2).
- Logistic Regression.
- Scikit-learn.

The classifier predicts one of three answer categories:

- Poor
- Average
- Good

**Important:** The ML classifier is experimental. Its prediction is stored separately from the main rubric-based evaluation score.

#### Experimental Model Results

The recorded experiment used a small illustrative dataset containing 57 examples across 19 prompt groups.

| Metric | Result |
|---|---:|
| Accuracy | 33.33% |
| Macro Precision | 13.89% |
| Macro Recall | 33.33% |
| Macro F1-score | 19.61% |

These results indicate that the current classifier requires significant improvement. They should not be considered evidence of reliable real-world interview assessment performance.

### 6. Voice-Related Components

The project includes components for:
- Browser speech recognition.
- Browser speech synthesis.
- Audio recording.
- Experimental speaker verification using SpeechBrain ECAPA-TDNN.

Actual browser compatibility and end-to-end functionality should be verified before treating these features as production-ready.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Django | Web application framework |
| SQLite | Development database |
| Scikit-learn | Experimental ML classifier |
| TF-IDF | Text feature extraction |
| Logistic Regression | Text classification |
| SpeechBrain | Experimental speaker verification |
| HTML and CSS | User interface |
| JavaScript | Browser-side functionality |
| JSON | Question bank data |
| uv | Python dependency management |

## Project Structure

```text
AI_Interview_System/
│
├── config/
│   └── Django project configuration
│
├── data/
│   └── question_bank.json
│
├── interviews/
│   ├── templates/
│   ├── evaluation_engine.py
│   ├── final_assessment_engine.py
│   ├── question_engine.py
│   ├── models.py
│   ├── views.py
│   └── tests.py
│
├── roles/
│
├── src/
│   └── ai_interview_system/
│
├── manage.py
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

*Note: Additional ML training, model, and migration files may exist in the latest project version.*

## Installation and Setup

### Requirements

- Python version supported by the project's `.python-version` file.
- uv package manager.
- Git.
- A Windows, Linux, or macOS development environment.

### Step 1: Clone the Repository

```bash
git clone https://github.com/miansemab/AI-Interview-System.git
```

### Step 2: Open the Project Directory

```bash
cd AI-Interview-System
```

### Step 3: Install Dependencies

```bash
uv sync
```

### Step 4: Apply Database Migrations

```bash
uv run python manage.py migrate
```

### Step 5: Populate the Question Bank

If the question bank has not already been populated:

```bash
uv run python manage.py seed_question_bank
```

### Step 6: Run the Development Server

```bash
uv run python manage.py runserver
```

Open the following address in your browser:

```text
http://127.0.0.1:8000/
```

## Machine Learning Model

The experimental classifier follows this workflow:

```text
User Answer
    |
    v
Text Preprocessing
    |
    v
TF-IDF Vectorization
    |
    v
Logistic Regression
    |
    v
Poor / Average / Good
```

The experimental prediction is stored separately from the authoritative rubric-based score.

## Testing

The project includes automated Django tests.

Run the tests using:

```bash
uv run python manage.py test
```

Check the Django project configuration:

```bash
uv run python manage.py check
```

Train the experimental answer classifier, if its training command and dependencies are available:

```bash
uv run python manage.py train_answer_classifier
```

## Current Limitations

- The experimental ML classifier has limited training data and low evaluation performance.
- More diverse and professionally labelled training examples are required.
- Voice functionality requires further browser and device testing.
- Speaker verification requires further validation.
- The application is a development-stage project and is not yet a production-ready interview assessment platform.

## Future Improvements

- Improve the ML classifier using a larger and better-labelled dataset.
- Explore advanced NLP and transformer-based models.
- Improve voice interaction and speech processing.
- Enhance speaker verification.
- Improve the user interface and user experience.
- Add more technical interview categories.
- Improve assessment reports and performance analytics.

## Academic Project Information

**Project Title:** Machine Learning Based Voice Interview Simulator

**Developer:** Mian Semab

**GitHub:** https://github.com/miansemab

**Repository:** https://github.com/miansemab/AI-Interview-System

## Acknowledgement

This project was developed as a practical exploration of Python, Django, software engineering, Machine Learning, text classification, and voice-related technologies.

Contributions, suggestions, and improvements are welcome.
