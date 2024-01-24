import csv
from flask import Blueprint, render_template, request, current_app, redirect, url_for
from io import StringIO
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from langchain.document_loaders import DataFrameLoader
from yellowbrick.cluster import KElbowVisualizer
from transformers import GPT2TokenizerFast
import time
from openai import OpenAI, OpenAIError
import numpy as np
import json
from ast import literal_eval
import re

bp = Blueprint("routes", __name__)

client = OpenAI(
  organization='org-YNDucfmZbZO9yaWvN2Ma6GUT',
  api_key="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"
)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

@bp.route("/upload", methods=["GET"])
def upload_form():
    return render_template("upload_form.html")

def split_docs(documents,chunk_size=5000,chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def complete_turbo(prompt_txt):
    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            res = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "user", "content": prompt_txt}
                ]
            )
            return str(res.choices[0].message.content.strip())
        except OpenAIError as e:
            print(f"An error occurred: {e}")
            retries += 1
            print(f"Retrying ({retries}/{max_retries})...")
            time.sleep(2 ** retries)  # Exponential backoff for waiting between retries

    return "Failed to Generate Response"

def combine_posts_within_limit(posts_df, limit):
    combined_text = ""
    total_tokens = 0
    first_snippet = True
    separator_length = len(tokenizer(" ### ")['input_ids'])

    for snippet in posts_df['content']:
        # Calculate the tokens of the current snippet
        snippet_tokens = len(tokenizer(snippet)['input_ids'])

        # Check if adding the next snippet with separator exceeds token limit
        if not first_snippet and total_tokens + snippet_tokens + separator_length > limit:
            break

        # Add the snippet with separator to the combined text
        if not first_snippet:
            combined_text += " ### "
            total_tokens += separator_length
        combined_text += snippet
        total_tokens += snippet_tokens

        first_snippet = False
    return combined_text


def format_to_clustered_json(df):
    # Grouping by 'cluster' and aggregating other columns
    grouped = df.groupby(['cluster', 'cluster_summary', 'cluster_title']).agg(lambda x: x.tolist()).reset_index()

    # Merging cluster_keywords into a single list for each cluster
    grouped['cluster_keywords'] = grouped['cluster_keywords'].apply(lambda x: list(set([item for sublist in x for item in sublist])))

    # Creating a structure based on the specified format
    clusters = []
    for _, row in grouped.iterrows():
        cluster_data = {
            "cluster": row["cluster"],
            "summary": row["cluster_summary"],
            "title": row["cluster_title"],
            "keywords": row["cluster_keywords"],
            "posts": []
        }

        for i in range(len(row['alert_id'])):
            alert_obj = {key: row[key][i] for key in df.columns if key not in ['cluster', 'cluster_summary', 'cluster_title', 'cluster_keywords']}
            cluster_data["posts"].append(alert_obj)

        clusters.append(cluster_data)

    # Converting the clusters list to JSON
    #json_output = json.dumps(clusters, indent=4)
    return clusters


def load_dataframe_into_langchain(df):
      loader = DataFrameLoader(df, page_content_column="combined")
      return loader.load()


def run_ai(input_csv):
    # Import CSV
    data = pd.DataFrame(input_csv)
    data = data.dropna()
    data.rename(columns=lambda x: x.strip().lower(), inplace=True)

    # Combine data
    data["combined"] = (
        "Date: " + data.date.str.strip() + "; Alert Type: " + data.alert_type.str.strip() + "; Title: " + data.title.str.strip() + "; Content: " + data.content.str.strip() + "; Link: " + data.link.str.strip() + "; Location: " + data.location.str.strip() 
    )    

    documents = load_dataframe_into_langchain(data)

    # Setup Chunking algorithm
    docs = split_docs(documents)

    # Setup Embedding Client
    embedding_function = OpenAIEmbeddings(openai_api_key="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", model="text-embedding-ada-002")

    # Load docs into chromadb
    db = Chroma.from_documents(docs, embedding_function)

    # Load embeddings
    vectorstore_data = db.get(include=["embeddings", "metadatas"])
    embeddings = vectorstore_data["embeddings"]

    # Setup Elbow method
    matrix = np.vstack(embeddings)

    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    vis_dims = tsne.fit_transform(matrix)

    # Elbow method
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2,16), timings= True)
    visualizer.fit(matrix)        # Fit data to visualizer

    # K means clustering
    n_clusters = visualizer.elbow_value_
    # Add more clusters
    kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=60)
    kmeans.fit(matrix)
    kmeans_labels = kmeans.labels_
    kmeans_labels = kmeans_labels[:-1]
    data['cluster'] = kmeans_labels

    # Generating Summaries
    sum_data = pd.DataFrame(columns=['cluster', 'cluster_summary', 'cluster_title', 'cluster_keywords'])

    for i in range(n_clusters):
        #print(f"Cluster {i} Summary:", end=" ")

        # Combine social media posts within limit
        combined_posts = combine_posts_within_limit(data[data.cluster == i], 15000)

        # Generate Sumamry
        response = complete_turbo(f'Write a paragraph summary and include the common theme of the News Stories below, find simmilarities and describe the details of the news stories and classify the overall sentiment of the news stories. Make sure the summary is one snippet without breaks. \n\nNews Story:\n"""\n{combined_posts}\n"""\n\nSummary:')
        # Generate Title
        response_title = complete_turbo(f'Create a short Title for the common themes or topics in the news story below, make sure the title is not longer than 10 words, just include the title in the output. \n\nNews Story:\n"""\n{combined_posts}\n"""\n\nTitle:')
        # Generate Keywords
        response_keywords = complete_turbo(f'Generate 2-8 keywords not longer that one word per keyword for the news stories below, make sure the keywords are in english even if the stories are in a different language, make sure the keywords are only generated in the following format and only include the following no other text and make sure to only include one python array or keywords: ["keyword", "keyword2", "keyword3"]. \n\nNews Story:\n"""\n{combined_posts}\n"""\n\nKeywords:')

        llm_keywords = literal_eval(response_keywords.strip())

        #print("Sum", response)

        # Check Token Length of generated summary (Testing)
        #print("\nTitle = ", response_title)

        # Check Token Length of generated summary (Testing)
        #print("\nPosts Token Length = ",len(tokenizer(combined_posts)['input_ids']))

        # Append Summary to DF
        line = pd.DataFrame([[i, response.replace("\n", ""), response_title, llm_keywords]], columns=['cluster', 'cluster_summary', 'cluster_title', 'cluster_keywords'])
        sum_data = pd.concat([line, sum_data], ignore_index=True)

        # Print tweets that made up summary (Testing)
        #print("\n")
        sample_cluster_rows = data[data.cluster == i].sample(30, replace=True, random_state=42)
    
    # Merge Summaries + titles to main df
    data=data.merge(sum_data, on=['cluster'],how='left')

    # Formatting DataFrame to JSON as per the specified structure
    formatted_json = format_to_clustered_json(data)
    
    return formatted_json


@bp.route("/upload", methods=["POST"])
def upload():
    text_body = request.form.get("text_body", "")

    if "csv_file" not in request.files:
        return "No file part"

    csv_file = request.files["csv_file"]

    if csv_file:
        data = parse_csv(csv_file)
        current_app.logger.info(f"CSV Data: {data}")

        response = run_ai(data)

        dummy_data = [
            {
                "title": "Article 1",
                "summary": "Summary 1",
                "keywords": ["keyword1", "keyword2"],
                "posts": [
                    {
                        "date": "2023-11-17",
                        "time": "14:30",
                        "source": "Twitter",
                        "location": "New York, NY",
                        "account_name": "user123",
                        "post_title": "Exciting News!",
                        "body": "Just announced a new project. Stay tuned for details!",
                        "hashtag": ["#announcement", "#newproject"],
                        "video_available": True,
                        "cluster": "Technology",
                        "summary": "New project announcement in the technology sector.",
                        "title": "Project Announcement",
                        "keywords": ["technology", "announcement", "project"],
                    },
                    {
                        "date": "2023-11-17",
                        "time": "14:30",
                        "source": "Twitter",
                        "location": "New York, NY",
                        "account_name": "user123",
                        "post_title": "Exciting News!",
                        "body": "Just announced a new project. Stay tuned for details!",
                        "hashtag": ["#announcement", "#newproject"],
                        "video_available": True,
                        "cluster": "Technology",
                        "summary": "New project announcement in the technology sector.",
                        "title": "Project Announcement",
                        "keywords": ["technology", "announcement", "project"],
                    },
                    {
                        "date": "2023-11-17",
                        "time": "14:30",
                        "source": "Twitter",
                        "location": "New York, NY",
                        "account_name": "user123",
                        "post_title": "Exciting News!",
                        "body": "Just announced a new project. Stay tuned for details!",
                        "hashtag": ["#announcement", "#newproject"],
                        "video_available": True,
                        "cluster": "Technology",
                        "summary": "New project announcement in the technology sector.",
                        "title": "Project Announcement",
                        "keywords": ["technology", "announcement", "project"],
                    },
                ],
            },
            {
                "title": "Article 2",
                "summary": "Summary 2",
                "keywords": ["keyword2", "keyword2"],
                "posts": [
                    {
                        "date": "2023-11-17",
                        "time": "14:30",
                        "source": "Twitter",
                        "location": "New York, NY",
                        "account_name": "user123",
                        "post_title": "Exciting News!",
                        "body": "Just announced a new project. Stay tuned for details!",
                        "hashtag": ["#announcement", "#newproject"],
                        "video_available": True,
                        "cluster": "Technology",
                        "summary": "New project announcement in the technology sector.",
                        "title": "Project Announcement",
                        "keywords": ["technology", "announcement", "project"],
                    },
                    {
                        "date": "2023-11-17",
                        "time": "14:30",
                        "source": "Twitter",
                        "location": "New York, NY",
                        "account_name": "user123",
                        "post_title": "Exciting News!",
                        "body": "Just announced a new project. Stay tuned for details!",
                        "hashtag": ["#announcement", "#newproject"],
                        "video_available": True,
                        "cluster": "Technology",
                        "summary": "New project announcement in the technology sector.",
                        "title": "Project Announcement",
                        "keywords": ["technology", "announcement", "project"],
                    },
                ],
            }
        ]

        return render_template("upload_form.html", text_body=text_body, data=response)

    return redirect(url_for("routes.upload_form"))


def parse_csv(csv_file):
    csv_data = []
    csv_stream = StringIO(csv_file.stream.read().decode("UTF-8"))
    csv_reader = csv.DictReader(csv_stream)
    for row in csv_reader:
        csv_data.append(row)
    return csv_data