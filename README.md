# NLP Task Extraction and Categorization Pipeline

## Overview
This project implements an NLP pipeline to extract tasks from unstructured text and categorize them into meaningful groups. It uses SpaCy for basic NLP tasks and K-means clustering with TF-IDF embeddings for categorization. No large language models (LLMs) or annotated datasets are used, adhering to assignment constraints.

## Objective
- **Extract tasks/action items** from a paragraph:
  - **Mandatory**: Identify the action to be performed.
  - **Good to Have**: Identify the responsible person and deadline (if present).
- **Categorize tasks** into useful groups (e.g., "Errands," "Chores").

## File Structure
```
NLP_Pipeline/
│── packages.py   # Sets up dependencies (install/import required libraries)
│── main.py       # Main program for task extraction and categorization
│── README.md     # Project documentation
```

## Requirements
- **Python 3.7+**
- Install required dependencies by running `packages.py`

### Required Libraries:
- `spacy` (with `en_core_web_sm` model)
- `scikit-learn`
- `numpy`
- `json`

## Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/NLP_Pipeline.git
   cd NLP_Pipeline
   ```

2. **Run `packages.py` First**
   ```bash
   python packages.py
   ```
   - This will install and import all necessary dependencies.

3. **Restart the Kernel/Session**
   - If using **Jupyter Notebook**, click **Kernel → Restart Kernel**.
   - If using **Google Colab**, click **Runtime → Restart Runtime**.
   - If using **a local Python script**, close and reopen the terminal.

4. **Run `main.py`**
   ```bash
   python main.py
   ```
   - This will process the input text and output the extracted tasks in JSON format.

## Implementation Details

### Steps
1. **Preprocessing (`clean_text()`)**
   - Removes special characters (except `.,!?`) and normalizes spaces.
   - Stop word removal is **disabled** to preserve task context.

2. **Task Extraction (`extract_tasks()`)**
   - Identifies sentences with **root verbs** and **task-related keywords** (e.g., "buy," "clean," "should").
   - Extracts persons using:
     - Named Entity Recognition (NER)
     - Proper nouns
     - Pronoun resolution (e.g., `"He"` → `"Rahul"`).
   - Detects deadlines using SpaCy NER and regex (e.g., `"by 5 pm"`, `"tomorrow"`).

3. **Categorization (`generate_embeddings()` & `categorize_tasks()`)**
   - Generates **TF-IDF embeddings** for task sentences.
   - Clusters tasks using **K-means** (`max_clusters=4`).
   - Aligns clusters to categories (**"Chores," "Errands," "Work/Studies," "Meetings"**) using heuristic rules.

### Sample Input:
```python
text = """Rahul wakes up early every day. He goes to college in the morning and comes back at 3 pm.
At present, Rahul is outside. He has to buy the snacks for all of us. Rahul should clean the room by 5 pm today."""
```

### Sample Output:
```json
{
    "tasks": [
        {
            "task": "He has to buy the snacks for all of us.",
            "person": "Rahul",
            "deadline": null,
            "category": "Errands"
        },
        {
            "task": "Rahul should clean the room by 5 pm today.",
            "person": "Rahul",
            "deadline": "5 pm today",
            "category": "Chores"
        }
    ],
    "embeddings": [...]
}
```

## Validation
- **Manual validation** compared computed tasks against expected results:
  - ✅ Task 1: `"He has to buy..."` → Matched (person: `"Rahul"`, deadline: `null`, category: `"Errands"`).
  - ✅ Task 2: `"Rahul should clean..."` → Matched (person: `"Rahul"`, deadline: `"5 pm today"`, category: `"Chores"`).

