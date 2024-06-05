from flask import Flask, request, jsonify, render_template, send_file
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

app = Flask(__name__)

# Load the pre-trained models and data
cv = joblib.load('count_vectorizer.pkl')
bow_matrix = joblib.load('bow_matrix.pkl')
df_nlp = pd.read_csv('df_nlp.csv', sep='\t')  # Ensure the correct separator is used
sel_df = pd.read_csv('sel_df2.csv', sep='\t')  # Load your selection dataframe

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    text = ' '.join(tokens)
    return text.lower()

def make_json_serializable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.where(pd.notnull(obj), None).to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.where(pd.notnull(obj), None).to_dict()
    else:
        return obj

@app.route('/')
def index():
    return render_template('index.html')
    # return send_file('src/html/index.html')

@app.route('/ui-buttons')
def ui_buttons():
    return render_template('ui-buttons.html')

@app.route('/artikel')
def artikel():
    return render_template('artikel.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # Preprocess and transform the input text
    clean_text_input = clean_text(text)
    new_text_bow = cv.transform([clean_text_input]).toarray()

    # Calculate similarity
    similarity_scores = cosine_similarity(bow_matrix, new_text_bow)

    # Find the cluster with highest similarity for the input text
    predicted_cluster_index = similarity_scores[:, 0].argmax()
    predicted_industry = df_nlp.iloc[predicted_cluster_index]['industry']
    predicted_skill = df_nlp.iloc[predicted_cluster_index]['nama_kemampuan']
    predicted_pengalaman = df_nlp.iloc[predicted_cluster_index]['tingkat_pengalaman_terformat']
    predicted_clusters = df_nlp.iloc[predicted_cluster_index]['Clusters']
    predicted_course = df_nlp.iloc[predicted_cluster_index]['course']

    cluster_message = ""
    if predicted_clusters == 0:
        cluster_message = "Startup atau SMEs di bidang pendidikan dan pelatihan."
    elif predicted_clusters == 1:
        cluster_message = "Corporate besar atau multinational companies di bidang keuangan."
    elif predicted_clusters == 2:
        cluster_message = "Corporate besar atau perusahaan teknologi besar."
    elif predicted_clusters == 3:
        cluster_message = "Startup atau SMEs di bidang kesehatan dan Teknologi."
    else:
        cluster_message = "Nilai prediksi cluster di luar rentang yang diharapkan."

    # Find the top 3 job IDs
    top_3_indices = similarity_scores[:, 0].argsort()[-3:][::-1]
    top_3_job_ids = df_nlp.iloc[top_3_indices]['id_pekerjaan'].values.tolist()

    # Filter sel_df based on predicted job IDs
    filtered_sel_df = sel_df[sel_df['id_pekerjaan'].isin(top_3_job_ids)].drop_duplicates(subset='id_pekerjaan')

    # Convert the filtered dataframe to a JSON-serializable format
    filtered_jobs = make_json_serializable(filtered_sel_df)

    response = {
        'industry': predicted_industry,
        'skill': predicted_skill,
        'experience_level': predicted_pengalaman,
        'clusters': str(predicted_clusters),
        'course': predicted_course,
        'cluster_message': cluster_message,
        'filtered_jobs': filtered_jobs
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)