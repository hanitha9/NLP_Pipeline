# NLP Assignment - Part A: Task Extraction and Categorization

## Overview
This project implements an **NLP pipeline** to extract tasks from unstructured text and categorize them into meaningful groups. The solution uses a heuristic-based approach with **SpaCy for NLP tasks** and **K-means clustering with TF-IDF embeddings** for categorization. The implementation avoids large language models (LLMs) or annotated datasets, following assignment constraints.

## Objective
- Extract tasks/action items from a paragraph, including:
  - **Mandatory:** The action to be performed.
  - **Good to Have:** The person responsible and deadline (if present).
- Categorize tasks into useful groups (e.g., "Errands," "Chores").

## Requirements
- **Python 3.7+**
- **Required Libraries:**
  - `spacy` (with `en_core_web_sm` model)
  - `scikit-learn`
  - `numpy`
  - `json`
- Install dependencies using:
  ```bash
  pip install spacy scikit-learn numpy
  python -m spacy download en_core_web_sm
  ```

## File Structure
```
├── part_a_solution.py  # Main Python script containing the complete solution
├── README.md           # Project documentation
```

## How to Run
1. **Clone or Download** the repository.
2. **Save `part_a_solution.py`** to your local machine.
3. **Install Dependencies** using the commands mentioned above.
4. **Execute the script**:
   ```bash
   python part_a_solution.py
   ```
   The script processes a sample input and outputs the results in **JSON format**, followed by validation and insights.

## Implementation Details

### Steps
1. **Preprocessing (`clean_text()`)**
   - Removes special characters (except `.`, `,`, `!`, `?`) and normalizes spaces.
   - Stop word removal is **disabled** to preserve key task-related words.

2. **Task Extraction (`extract_tasks()`)**
   - Uses heuristics: Identifies sentences with **root verbs** and task-related keywords (e.g., "buy," "clean," "should").
   - Extracts persons via **NER, proper nouns, or pronoun resolution** (e.g., *"He" → "Rahul"*).
   - Detects deadlines with **SpaCy NER and regex** (e.g., "by 5 pm," "tomorrow").

3. **Categorization (`generate_embeddings()` & `categorize_tasks()`)**
   - Generates **TF-IDF embeddings** for task sentences.
   - **Clusters tasks using K-means** (max 4 clusters).
   - Aligns clusters to **predefined categories** ("Chores," "Errands," "Work/Studies," "Meetings") using heuristics.

## Sample Input
```python
text = """Rahul wakes up early every day. He goes to college in the morning and comes back at 3 pm.
At present, Rahul is outside. He has to buy the snacks for all of us. Rahul should clean the room by 5 pm today."""
```

## Sample Output
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
### **Manual Validation:**
Computed tasks were manually validated against a curated sample:
| Task | Extracted Person | Extracted Deadline | Assigned Category |
|------|-----------------|---------------------|------------------|
| "He has to buy..." | Rahul | None | Errands |
| "Rahul should clean..." | Rahul | 5 pm today | Chores |
✅ **Result: Both tasks match perfectly.**

## Insights and Challenges
### **Key Learnings:**
- **Stop Word Removal:** Enabling it removed important words like "has," which **broke task detection**. It was **disabled** for accuracy.
- **Category Swapping:** Initial **K-means clustering** incorrectly categorized tasks (e.g., *"buy" as "Chores"*). A heuristic **fixed misalignments**.
- **Pronoun Resolution:** Assigning pronouns like *"He" → "Rahul"* was tricky but worked using a **last_person variable**.
- **Balancing Preprocessing:** Removing too much noise **disrupted task extraction**, requiring careful tuning.

---

### 🔗 **Future Improvements**
- Extend **NER-based categorization** (e.g., detecting meeting-related terms like "call," "schedule").
- Improve **deadline parsing** with `dateparser` for better handling of "tomorrow," "next Monday," etc.
- Experiment with **BERT embeddings** for better **task clustering** (if LLMs are allowed in future versions).

---

### 🔥 **Contributions & Issues**
- Feel free to **fork this repo**, report issues, or suggest improvements via **Pull Requests**.
- If you find this project helpful, consider **starring ⭐ the repo**!

---

🚀 **Happy Coding!** 🎯

