# %%
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/UniWor/TCD/NLP/sentiment_analysis/translation-project-455814-06061264bdb5.json"
from google.cloud import translate_v3 as translate
import time
import pandas as pd
from srbai.Alati.Transliterator import transliterate_cir2lat

# %%
# Load the English movie review dataset
imdb_data = pd.read_csv('IMDB.csv')

# %%
# Sample the dataset to match the size of the Serbian dataset
positive_reviews = imdb_data[imdb_data['sentiment'] == 'positive'].sample(n=814, random_state=42)
negative_reviews = imdb_data[imdb_data['sentiment'] == 'negative'].sample(n=814, random_state=42)

balanced_imdb = pd.concat([positive_reviews, negative_reviews]).reset_index(drop=True)

balanced_imdb.to_csv('Balanced_IMDB.csv', index=False)

# %%
imdb_data = pd.read_csv('Balanced_IMDB.csv')

# %%
# Google Cloud Project ID
PROJECT_ID = "translation-project-455814"
location = "global"
parent = f"projects/{PROJECT_ID}/locations/{location}"

client = translate.TranslationServiceClient()

def translate_text(text: str, source_language_code: str = "en", target_language_code: str = "sr-Latn") -> str:
    try:
        response = client.translate_text(
            parent=parent,
            contents=[text],
            mime_type="text/plain",
            source_language_code=source_language_code,
            target_language_code=target_language_code,
        )
        translated_text = response.translations[0].translated_text
        return translated_text

    except Exception as e:
        print(f"Translation failed: {e}")
        return ""

# %%
# Translate the reviews and replace the original reviews
translated_reviews = []

for index, row in imdb_data.iterrows():
    if index % 50 == 0:
        print(f"Translating review {index}/{len(imdb_data)}...")
    
    translated_review = translate_text(row['review'], source_language_code='en', target_language_code='sr-Latn')
    translated_reviews.append(translated_review)
    
    # A slight delay was needed to avoid rate-limiting
    time.sleep(0.2)

imdb_data['review'] = translated_reviews
imdb_data.to_csv('Translated_IMDB_Latin.csv', index=False)



# %%
# Load the translated dataset
imdb_data = pd.read_csv('Translated_IMDB_Latin.csv', encoding='utf-8')
# Apply the transliterator to the 'review' column
imdb_data['review'] = imdb_data['review'].apply(lambda text: transliterate_cir2lat(text) if isinstance(text, str) else text)
imdb_data.to_csv('Properly_Romanized_IMDB.csv', index=False, encoding='utf-8')

# %%
# Check the final result since excel doesn't display this alphabet properly
imdb_data = pd.read_csv('Properly_Romanized_IMDB.csv', encoding='utf-8')
print(imdb_data.head())


# %%



