import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
#Bai 1
# 1
df = pd.read_csv(r'C:\LuyenTap\Labs\Lab4\ITA105_Lab_4_Hotel_reviews.csv') 
df['review_text'] = df['review_text'].fillna('')
df['hotel_name'] = df['hotel_name'].fillna('Unknown')
print(df.isnull().sum())

# 2 
le = LabelEncoder()
df['hotel_name_encoded'] = le.fit_transform(df['hotel_name'])
df['customer_type_encoded'] = le.fit_transform(df['customer_type'])

# 3
stop_words = ['và', 'của', 'là', 'nhưng', 'có', 'cho', 'với']

def preprocess_text(text):
     
    text = text.lower()
     
    text = re.sub(r'[^\w\s]', '', text)
     
    tokens = text.split()
     
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

df['clean_review'] = df['review_text'].apply(preprocess_text)

# 4 
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['clean_review'])
print(f"\nKích thước ma trận TF-IDF: {tfidf_matrix.shape}")

# 5 
sentences = [row.split() for row in df['clean_review']]

 
model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
  
try:
    similar_words = model_w2v.wv.most_similar('sạch', topn=5)
    print("\n5 từ gần nghĩa nhất với 'sạch':")
    for word, score in similar_words:
        print(f"- {word}: {score:.4f}")
except KeyError:
    print("\nTừ 'sạch' không xuất hiện đủ số lần trong tập dữ liệu nhỏ này.")

 
print("\nDữ liệu sau khi Encoding (5 dòng đầu):")
print(df[['hotel_name', 'hotel_name_encoded', 'customer_type', 'customer_type_encoded']].head())

#Bai 2
#1
df = pd.read_csv(r'C:\LuyenTap\Labs\Lab4\ITA105_Lab_4_Match_comments.csv')

df['team'] = df['team'].fillna('Unknown')
df['comment_text'] = df['comment_text'].fillna('')

# 2 
le = LabelEncoder()
df['team_encoded'] = le.fit_transform(df['team'])
df['author_encoded'] = le.fit_transform(df['author'])

# 3 
vietnamese_stopwords = ['và', 'của', 'là', 'nhưng', 'có', 'cho', 'với', 'nhưng']

def clean_comment(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Xóa dấu câu
    tokens = text.split()
    tokens = [t for t in tokens if t not in vietnamese_stopwords]
    return " ".join(tokens)

df['clean_comment'] = df['comment_text'].apply(clean_comment)

# 4 
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['clean_comment'])
print(f"Kích thước ma trận TF-IDF: {tfidf_matrix.shape}")

# 5 
sentences = [row.split() for row in df['clean_comment']]
model_w2v = Word2Vec(sentences, vector_size=50, window=3, min_count=1, workers=4)

try:
    similar_words = model_w2v.wv.most_similar('xuất', topn=5)
    print("\n5 từ gần nghĩa nhất với 'xuất sắc' (dựa trên từ 'xuất'):")
    for word, score in similar_words:
        print(f"- {word}: {score:.4f}")
except KeyError:
    print("\nTừ khóa chưa xuất hiện đủ để huấn luyện.")

# 6 
print("\nBảng mã hóa Team và Author (5 dòng đầu):")
print(df[['team', 'team_encoded', 'author', 'author_encoded']].head())

#Bai 3
# 1 
df = pd.read_csv(r'C:\LuyenTap\Labs\Lab4\ITA105_Lab_4_Player_feedback.csv')

 
df['device'] = df['device'].fillna('unknown')
df['feedback_text'] = df['feedback_text'].fillna('')

# 2 
le = LabelEncoder()
df['player_type_encoded'] = le.fit_transform(df['player_type'])
df['device_encoded'] = le.fit_transform(df['device'])

# 3 
stop_words = ['nhưng', 'và', 'là', 'của', 'có', 'cho', 'với', 'hơi']

def preprocess_feedback(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

df['clean_feedback'] = df['feedback_text'].apply(preprocess_feedback)

# 4 
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['clean_feedback'])
print(f"Kích thước ma trận TF-IDF: {tfidf_matrix.shape}")

# 5 
sentences = [row.split() for row in df['clean_feedback']]
 
model_w2v = Word2Vec(sentences, vector_size=64, window=5, min_count=1, workers=4)

try:
    similar_words = model_w2v.wv.most_similar('đẹp', topn=5)
    print("\n5 từ gần nghĩa nhất với 'đẹp':")
    for word, score in similar_words:
        print(f"- {word}: {score:.4f}")
except KeyError:
    print("\nTừ 'đẹp' không xuất hiện đủ để tìm từ gần nghĩa.")

  
print("\nDữ liệu sau khi xử lý (5 dòng đầu):")
print(df[['player_type', 'player_type_encoded', 'device', 'device_encoded', 'clean_feedback']].head())

#Bai 4
#1
df = pd.read_csv(r'C:\LuyenTap\Labs\Lab4\ITA105_Lab_4_Album_reviews.csv')

 
df['review_text'] = df['review_text'].fillna('')
df['genre'] = df['genre'].fillna('Unknown')
df['platform'] = df['platform'].fillna('Unknown')

# 2 
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['genre'])
df['platform_encoded'] = le.fit_transform(df['platform'])

print("--- Danh mục sau khi Encoding ---")
print(df[['genre', 'genre_encoded', 'platform', 'platform_encoded']].head())

# 3 
stop_words = ['và', 'của', 'là', 'nhưng', 'có', 'cho', 'với', 'rất', 'cũng']

def preprocess_review(text):
   
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', '', text)
   
    tokens = text.split()
    
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

df['clean_review'] = df['review_text'].apply(preprocess_review)

# 4 
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['clean_review'])
print(f"\nKích thước ma trận TF-IDF: {tfidf_matrix.shape}")

# 5  
sentences = [row.split() for row in df['clean_review']]

model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

try:
    similar_words = model_w2v.wv.most_similar('sáng', topn=5)
    print("\n5 từ gần nghĩa nhất với 'sáng tạo' (dựa trên token 'sáng'):")
    for word, score in similar_words:
        print(f"- {word}: {score:.4f}")
except KeyError:
    print("\nTừ 'sáng' không có đủ dữ liệu để học trong tập này.")

print("\nKết quả xử lý cuối cùng (5 dòng đầu):")
print(df[['review_text', 'clean_review', 'rating']].head())