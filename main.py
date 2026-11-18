import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({
    "title": ["Space Journey", "Galaxy Wars", "Love in Paris", "The Last Detective",
              "Robot Future", "A Summer Romance", "Mystery at Midnight", "Planet Explorers"],
    "genres": ["science fiction adventure space", "science fiction action space",
               "romance drama paris", "crime mystery detective",
               "science fiction robots technology", "romance comedy summer",
               "mystery thriller crime", "adventure science fiction space"]
})
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(movies["genres"])
similarity = cosine_similarity(vectors)

def recommend(title, count=3):
    found = movies.index[movies.title.str.lower() == title.lower()].tolist()
    if not found:
        print("Unknown title. Available:", ", ".join(movies.title))
        return
    idx = found[0]
    ranked = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    print("Recommendations for", title)
    shown = 0
    for movie_idx, score in ranked:
        if movie_idx == idx:
            continue
        print(f"- {movies.iloc[movie_idx].title} (similarity: {score:.2f})")
        shown += 1
        if shown >= count:
            break

print("Movie Recommendation System")
print("Available:", ", ".join(movies.title))
recommend("Space Journey")
