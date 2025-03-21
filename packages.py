# Downgrade numpy to a compatible version
!pip install numpy==1.26.4

# Reinstall gensim to ensure compatibility
!pip install --upgrade gensim

# Install other required libraries
!pip install nltk spacy scikit-learn

# Download NLTK resources
import nltk
nltk.download('punkt')  # Fixed typo
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab') # Fixed typo

# Download SpaCy English model
!python -m spacy download en_core_web_sm

# Print instructions to restart the runtime
print("Please restart the runtime to load the SpaCy model. Then, re-run the code from the import section below.")
