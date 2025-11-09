# Project Documentation

This README provides instructions on how to set up and run the two projects:

1.  **Historical Events Timeline** (DBPedia based)
2.  **POS Tag Comparator** (NLP based)

## Prerequisites

- Python 3.x installed
- pip (Python package installer)

## Setup Instructions

Follow these steps to set up your development environment:

### 1. Create and Activate Virtual Environment

It is recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
```

Activate the virtual environment:

**On Windows:**
```bash
.\venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### 2. Install Dependencies

Once the virtual environment is activated, install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

## Project 1: Historical Events Timeline

**Goal:** Visualize 100 years of wars, inventions, births using DBPedia data.

To run this project, execute the `timeline.py` script:

```bash
python timeline.py
```

This script will generate an HTML file (e.g., `historical_timeline.html`) with the timeline visualization.

## Project 2: POS Tag Comparator (NLTK vs spaCy)

**Objective:** Compare POS tagging accuracy between NLTK and spaCy using the Brown Corpus.

To run this project, execute the `nlp_proj.py` script:

```bash
python nlp_proj.py
```

This script will perform the POS tagging comparison and output the results to the console.
