

# Project Title

Group members: Buket Sak, Anna Werner, Zeyuan Yu

## Introduction

TBD

## Dataset

This corpus contains preprocessed posts from the Reddit dataset (Webis-TLDR-17). The dataset consists of 3,848,330 posts with an average length of 270 words for content, and 28 words for the summary.

Features includes strings: author, body, normalizedBody, content, summary, subreddit, subreddit_id. Content is used as document and summary is used as summary.

Our research primarily focus on gaming subreddits including 'leagueoflegends', 'pokemon', 'zelda', 'Overwatch', 'smashbros'and 'hearthstone' and they have relatively large amount of post to research on.

| Subreddit          | Count   |
|--------------------|---------|
| leagueoflegends    | 109,307 |
| hearthstone        | 9,500   |
| pokemon            | 6,464   |
| smashbros          | 4,464   |
| Overwatch          | 3,633   |
| zelda              | 1,182   |

## Methods

### Preprocessing
Firstly, we used the content text that doesn't include TL;DR part because it could bias the phrase extraction towards words that are common in summaries. Summaries often repeat key points in a condensed way rather than reflecting the natural flow of the document.

We aimed to extract distinct phrases for each subreddit using a TF-IDF model. To improve the quality of the results, we first removed elements that are not useful for our task, such as links and stop words. In addition to stop words, we noticed that some phrases still appeared as important — for example, phrases like “quick play”, “ranked games”, or “playing overwatch”. To address this, we also removed certain domain-related words like “play”, “playing”, “game”, as well as the names of the games themselves, like "pokemon", "zelda", and "smash" etc., so that phrases containing these words would not dominate the results. Finally, we applied stemming to normalize word forms and avoid treating different surface forms of the same concept as distinct phrases.




```python
common_words = {"game", "games", "gaming", "player", "play", "playing", "plays", "pokemon", "zelda", "smash", "smashbros" "overwatch", "hearthstone", "legends"}
    
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text) #removes links if any
    text = re.sub(r'[^a-z\s]', ' ', text) #replaces anything that is not a letter or space with a space
    text = re.sub(r'\s+', ' ', text).strip() #replace multiple whitespace characters with a single space
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and word not in common_words]
    return " ".join(tokens)

# Apply to the filtered gaming dataset
df_games['cleaned_body'] = df_games['content'].apply(preprocess)
```


```python
df_games[['subreddit', 'body', 'cleaned_body']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subreddit</th>
      <th>body</th>
      <th>cleaned_body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>leagueoflegends</td>
      <td>Didn't they lose 6 games in a row? Just becaus...</td>
      <td>lose row close mean lot weaker team love arkan...</td>
    </tr>
    <tr>
      <th>64</th>
      <td>leagueoflegends</td>
      <td>Because spinning axe is a massive damage buff(...</td>
      <td>spinning axe massive damage buff percent based...</td>
    </tr>
    <tr>
      <th>106</th>
      <td>zelda</td>
      <td>My thoughts from nine months ago and its corre...</td>
      <td>thoughts nine months ago corresponding discuss...</td>
    </tr>
    <tr>
      <th>123</th>
      <td>leagueoflegends</td>
      <td>I agree, and feel like TRM...may or may not ge...</td>
      <td>agree feel like trm may may get lot unnecessar...</td>
    </tr>
    <tr>
      <th>124</th>
      <td>leagueoflegends</td>
      <td>I kinda of have issue with putting blame on ju...</td>
      <td>kinda issue putting blame one party yes agree ...</td>
    </tr>
  </tbody>
</table>
</div>



### Collecting Gensim's Phrases
Gensim's Phrases is a module designed for detecting and processing multi-word expressions within a text corpus, which is essential for enhancing natural language processing tasks. By identifying common phrases, such as "New York," as single entities rather than separate words, it improves the understanding of language context. The module employs statistical measures to evaluate the likelihood of word sequences forming phrases, allowing it to learn from the patterns present in the data. Once trained, the Phrases model can transform tokenized sentences, effectively combining recognized phrases into single tokens, which streamlines text preprocessing and enhances the performance of various NLP applications, including topic modeling and information retrieval. Overall, Gensim's Phrases significantly contributes to the quality and accuracy of text analysis by recognizing and processing complex language structures.
### Training Word2Vec
The Word2Vec model was trained on this phrased corpus (corpus_phrased), where common multi-word expressions were merged with underscores.

We used the skip-gram architecture to learn embeddings that predict surrounding context words.

The model had an embedding size of 100 dimensions, a window size of 5, ignored tokens that appeared fewer than 5 times, and was trained for 10 epochs.

### PCA
Principal Component Analysis (PCA)  (Bishop, 2006) is a powerful statistical technique widely used in document analysis to `reduce the dimensionality` of large datasets while preserving as much variance as possible Bishop. By transforming the original variables into a new set of uncorrelated variables called principal components, PCA helps in identifying patterns and structures within the data. This is particularly useful in text mining and natural language processing, where documents can be represented as high-dimensional vectors. By applying PCA, researchers can enhance the efficiency of various tasks such as clustering, classification, and visualization of textual data, ultimately leading to more insightful analyses.
### K-Means Clustering
K-Means clustering is a widely used unsupervised machine learning algorithm that partitions a dataset into distinct groups, or clusters, based on feature similarity. In the context of text analysis, K-Means is particularly effective for organizing large volumes of textual data, such as the posts from gaming subreddits analyzed in this report. By grouping similar documents, K-Means facilitates the identification of underlying patterns and themes within the data, enabling researchers to gain insights into user discussions and trends. The algorithm operates by iteratively assigning data points to the nearest cluster centroid and updating the centroids based on the mean of the assigned points, ultimately converging to a stable solution. This method is especially beneficial in natural language processing tasks, where it can enhance the understanding of complex language structures and improve the performance of various applications, including topic modeling and information retrieval (MacQueen, 1967).

### Emotion/Sentiment Classification
To quantify emotional responses in subreddit posts, particularly anger, we used pretrained transformer models from HuggingFace. These models allowed us to extract nuanced emotional and sentiment signals from Reddit comments.

#### Emotion Detection with DistilRoBERTa

We employed the `j-hartmann/emotion-english-distilroberta-base` model to classify each Reddit comment into one of seven emotion categories: anger, joy, sadness, fear, disgust, surprise, and neutral. The model outputs emotion probabilities, enabling us to use anger probability as a continuous dependent variable.

#### Why DistilRoBERTa?

- **Computational Efficiency**: DistilRoBERTa is computationally efficient while retaining ~97% of RoBERTa’s classification performance.
- **Emotion-specific Granularity**: It helps detect frustration, especially in multiplayer or competitive contexts.
- **Neutral Class Importance**: The neutral class is critical for distinguishing emotionally charged posts from mundane or off-topic ones.


### Setup 

We used Kaggle and Google Colab to conduct experiment with the dataset


### Experiments


Report how you conducted the experiments. We suggest including detailed explanations of the preprocessing steps and model training in your project. For the preprocessing, describe  data cleaning, normalization, or transformation steps you applied to prepare the dataset, along with the reasons for choosing these methods. In the section on model training, explain the methodologies and algorithms you used, detail the parameter settings and training protocols, and describe any measures taken to ensure the validity of the models.

#### PCA

Since the embeddings are high-dimensional, we applied Principal Component Analysis (PCA) to reduce them to 2 dimensions (X_reduced) for visualization.


```python
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import plotly.io as pio
```


```python
# Filter to only those in the vocab
valid_phrases = [p for p in unique_phrases if p in w2v_model.wv]

# Get their vectors
X = np.array([w2v_model.wv[p] for p in valid_phrases])

# Reduce to 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Set Plotly renderer
pio.renderers.default = 'notebook'

# Check data availability
#print(X_reduced.shape)
#print(valid_phrases)

# Create a DataFrame for Plotly
pca_df = pd.DataFrame({
    'x': X_reduced[:, 0],
    'y': X_reduced[:, 1],
    'phrase': valid_phrases
})

# Create interactive scatter plot
fig = px.scatter(
    pca_df, 
    x='x', 
    y='y', 
    text='phrase', #if this commented out, no text will show unless mouseover
    #hover_name='phrase',  # Show phrase on hover
    hover_data=['phrase'],  # Shows phrase on hover
    title="Interactive 2D PCA Visualization of Subreddit Bigrams"
)

# Adjust text position and size
fig.update_traces(
    textposition='top center',
    textfont_size=10,
    marker=dict(size=8)
)

# Add axis labels
fig.update_layout(
    xaxis_title="PCA 1",
    yaxis_title="PCA 2",
    hovermode='closest'
)

# Show the figure
fig.show(renderer='iframe')
```


<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_21.html"
    frameborder="0"
    allowfullscreen
></iframe>


<iframe src="{{ site.baseurl }}/figures/figure_21.html" width="100%" height="500"></iframe>

Phrases that appear close together likely have similar meanings or occur in similar contexts in subreddit data.

Phrases that are far apart are less related in terms of usage/context.

Since TfidfVectorizer was set to ngram_range=(2, 2), it only considered bigrams. As a result, all the top terms I extracted were bigrams in PCA. 
## Results and Discussion

Present the findings from your experiments, supported by visual or statistical evidence. Discuss how these results address your main research question.

## Conclusion

Summarize the major outcomes of your project, reflect on the research findings, and clearly state the conclusions you've drawn from the study.

## Contributions

| Team Member  | Contributions                                             |
|--------------|-----------------------------------------------------------|
| Alice Smith  | Data collection, preprocessing, model training, evaluation|                                                       |
| Bob Johnson  | ...                                                       |
| ...          | ...                                                       |

## References


1. Srinivasa-Desikan, Bhargav (2018). *Natural Language Processing and Computational Linguistics: A practical guide to text analysis with Python, Gensim, spaCy, and Keras*. Packt Publishing Ltd.
2. Bishop, Christopher M. (2006). *Pattern Recognition and Machine Learning (Information Science and Statistics)*. Springer-Verlag. Berlin, Heidelberg. ISBN: 0387310738.