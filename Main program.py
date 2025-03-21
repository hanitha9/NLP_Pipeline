import re
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import json

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text, remove_stop_words=False):
    """Clean text, optionally removing stop words."""
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    if remove_stop_words:
        doc = nlp(text)
        return ' '.join(token.text for token in doc if not token.is_stop)
    return text

def extract_tasks(text):
    """Extract tasks, persons, deadlines, and initial categories from text using heuristics."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    tasks = []
    last_person = None
    task_indicators = {"buy", "clean", "submit", "complete", "schedule", "review", "finish",
                       "should", "must", "need", "has", "have to", "do", "prepare", "call", 
                       "write", "send", "arrange", "fix", "organize"}

    for sent in sentences:
        doc_task = nlp(sent)
        tokens = [token.lemma_.lower() for token in doc_task]

        is_task = False
        root_verb = None
        for token in doc_task:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                root_verb = token.lemma_.lower()
                if any(indicator in tokens for indicator in task_indicators):
                    is_task = True
                break

        if is_task:
            task = sent
            person = None
            deadline = None
            category = "General"

            for ent in doc_task.ents:
                if ent.label_ == "PERSON":
                    person = ent.text
                    last_person = person
            if not person:
                for token in doc_task:
                    if token.pos_ == "PROPN" and token.dep_ in ["nsubj", "nsubjpass", "pobj"]:
                        person = token.text
                        last_person = person
                        break
            if not person:
                for token in doc_task:
                    if token.text.lower() in ["he", "she"] and last_person:
                        person = last_person
                        break
            if not person:
                for token in doc_task:
                    if token.dep_ in ["nsubj", "nsubjpass"] and token.pos_ in ["NOUN", "PROPN", "PRON"]:
                        person = token.text
                        if person.lower() in ["he", "she"] and last_person:
                            person = last_person
                        elif token.pos_ == "PROPN":
                            last_person = person
                        break

            for ent in doc_task.ents:
                if ent.label_ in ["TIME", "DATE"]:
                    deadline = ent.text
            if not deadline:
                deadline_match = re.search(r'(by\s+\d+\s*(am|pm)|today|tomorrow|before\s+\d+\s*(am|pm)|next\s+\w+|in\s+\d+\s*(hours|days)|at\s+\d+\s*(am|pm)|on\s+\w+)', 
                                         sent, re.IGNORECASE)
                if deadline_match:
                    deadline = deadline_match.group(0)

            tasks.append({"task": task, "person": person, "deadline": deadline, "category": category})
            print(f"Task Detected: {task}, Person: {person}, Deadline: {deadline}, Initial Category: {category}")
        else:
            for token in doc_task:
                if token.pos_ == "PROPN" and token.dep_ in ["nsubj", "nsubjpass", "pobj"]:
                    last_person = token.text
                    break
            print(f"Non-Task Sentence: {sent}, Last Person: {last_person}")

    return tasks

def generate_embeddings(sentences):
    """Generate TF-IDF embeddings for task sentences."""
    vectorizer = TfidfVectorizer(stop_words=None)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    return embeddings.tolist()

def categorize_tasks(tasks, embeddings, num_clusters=4):
    """Cluster tasks into categories using K-means, with heuristic alignment."""
    if not embeddings or len(embeddings) < 2:
        return ["Errands" if "buy" in task["task"].lower() else 
                "Chores" if "clean" in task["task"].lower() else 
                "Work/Studies" if "submit" in task["task"].lower() else 
                "Meetings" if "schedule" in task["task"].lower() else 
                "General" for task in tasks]
    
    kmeans = KMeans(n_clusters=min(num_clusters, len(embeddings)), random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    # Heuristic to align cluster labels with task content
    category_map = {}
    for i, task in enumerate(tasks):
        task_text = task["task"].lower()
        label = labels[i]
        if "buy" in task_text and "Errands" not in category_map.values():
            category_map[label] = "Errands"
        elif "clean" in task_text and "Chores" not in category_map.values():
            category_map[label] = "Chores"
        elif "submit" in task_text and "Work/Studies" not in category_map.values():
            category_map[label] = "Work/Studies"
        elif "schedule" in task_text and "Meetings" not in category_map.values():
            category_map[label] = "Meetings"
    
    # Fill remaining labels with unused categories
    all_categories = ["Chores", "Errands", "Work/Studies", "Meetings"]
    used_categories = set(category_map.values())
    remaining_categories = [cat for cat in all_categories if cat not in used_categories]
    for label in set(labels):
        if label not in category_map:
            category_map[label] = remaining_categories.pop(0)
    
    categories = [category_map[label] for label in labels]
    print(f"Cluster Labels: {labels}, Category Mapping: {category_map}, Assigned Categories: {categories}")
    return categories

# Example Input
text = """Rahul wakes up early every day. He goes to college in the morning and comes back at 3 pm.
At present, Rahul is outside. He has to buy the snacks for all of us. Rahul should clean the room by 5 pm today."""

# Processing Text
cleaned_text = clean_text(text, remove_stop_words=False)
tasks = extract_tasks(cleaned_text)
task_sentences = [task["task"] for task in tasks]
embeddings = generate_embeddings(task_sentences)
categories = categorize_tasks(tasks, embeddings)
for task, category in zip(tasks, categories):
    task["category"] = category

# Output Results
output = {"tasks": tasks, "embeddings": embeddings}
print(json.dumps(output, indent=4))
