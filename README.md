

# Project Title

Group members: Buket Sak, Anna Werner, Zeyuan Yu

## Introduction - P1

Online gaming communities have become vibrant spaces where players interact, share experiences, and form collective identities. Reddit, as a popular platform hosting a diverse range of gaming subreddits, offers a rich opportunity to study how language reflects and shapes these unique gaming cultures. Each subreddit not only discusses gameplay and strategies but also develops its own distinctive language patterns, emotional expressions, and thematic references that define its community.

Understanding how these communities use language—particularly distinctive phrases and emotional tone—can provide valuable insights into their cultural identities. Phrases frequently used within a subreddit often encode information about the game’s mechanics, characters, and themes, revealing what is important to that community. Moreover, emotional language patterns can reflect the social dynamics and shared attitudes prevalent among players.

This part of our project investigates whether distinct gaming communities on Reddit exhibit unique gaming cultural signatures. Our goal is to uncover semantic and emotional patterns that either differentiate or unite these communities.

To achieve this, we extract distinctive phrases using TF-IDF, train Word2Vec embeddings to capture semantic relationships within and across gaming communities, and use dimensionality reduction and clustering techniques to analyze the data. Additionally, we perform sentiment analysis on key phrases to explore variations in emotional tone across subreddits.

We examine five major gaming communities on Reddit: Hearthstone, League of Legends, Overwatch, Pokémon, Zelda, and Smashbros. We seek to answer the following questions for the first part of our project:

1. Do unique phrases reflect the specific themes, mechanics, or any characteristics of their respective games?

2. What insights can Word2Vec embeddings provide about the shared or distinct gaming culture in different subreddits based on phrases?

3. Do emotional patterns in phrase usage reveal shared or distinct elements across different gaming communities on Reddit?

Hyptheses: 

1. The most distinctive bigrams in each subreddit reflect the core game mechanics, characters, and thematic concerns unique to that gaming community, indicating that phrase usage is shaped by the content and culture of the game being discussed.
   
2. Word2Vec embeddings will capture meaningful semantic relationships between phrases, such that subreddits with overlapping gameplay styles will show similar phrase neighborhoods, while distinct communities will form separate clusters in the embedding space.

3. Game terms that share more features may reflect similar emotional patterns in phrase usage across communities.


## Introduction - P2

Previously, we conducted a TF-IDF analysis on several gaming-related subreddits to identify key terms and examined the sentiment of the sentences in which these terms appeared. This exploratory work suggested a connection between team play and emotional responses in gamer discussions. Building on this, and supported by existing research, we focus here on how anger expressions relate to game type, subreddit, and genre groups.

Meta-analyses (e.g., Anderson et al., 2010) have shown that violent video games increase aggression while reducing empathy and prosocial behavior.
This situates this project within the broader framework of the General Learning Model (GLM), as proposd by Buckley and Anderson (2006). The GLM posits that video game content strongly influences social behavior — violent games tend to increase aggression and reduce prosocial behavior, while prosocial games promote the opposite effects. These effects are bidirectional and can manifest both immediately and over long-term repeated exposure.

However, violent content is not the only factor driving anger. Smith et al. (2019) identify miscommunication and diverging goals within teams as internal stressors, while Behnke et al. (2021) report that weak teammates can elicit anger.

Because cooperation and empathy can counteract aggression (Eron & Huesmann, 1984), it is plausible that cooperative gameplay might reduce anger.

Motivated by these findings and the GLM framework, this project investigates whether cooperative gameplay moderates anger expressions in online gaming discourse. Specifically, we examine whether cooperative modes reduce anger compared to non-cooperative modes across different game types and subreddit contexts.

**Research Questions:**
1. Does team play increase anger expression in subreddit gaming communities?
2. Does cooperative gameplay reduce anger expression in gaming communities?

**Hypotheses:**

- H1: Greater team play (e.g., larger multiplayer modes) is associated with increased anger expression.
This reflects evidence that coordination failures and frustrating teammates often elicit anger, especially in competitive settings where performance matters.
- H2: Cooperative gameplay reduces anger expression.
Drawing on the GLM and research on prosocial games, cooperative play may foster empathy and social bonding, thereby buffering against anger—even in violent or competitive contexts.

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

#### PCA + K-means
Since the embeddings are high-dimensional, we applied Principal Component Analysis (PCA) to reduce them to 2 dimensions (X_reduced) for visualization.
KMeans groups phrases with similar vector representations.

- The model tries to group together bigrams that appeared in similar contexts across the data. For example, phrases about a particular game or character might cluster together because their usage patterns are similar.

- Each cluster may represent a different topic  (e.g., one cluster's tokens could be related to Overwatch, another to Pokémon, etc.).


```python
n_clusters = 6  # adjust this
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# Create DataFrame for plotting
pca_cluster_df = pd.DataFrame({
    'x': X_reduced[:, 0],
    'y': X_reduced[:, 1],
    'phrase': valid_phrases,
    'subreddit': subreddit_info,
    'cluster': labels.astype(str)  # Convert to string for nicer labels
})

# Plotly scatter plot
fig = px.scatter(
    pca_cluster_df,
    x='x',
    y='y',
    color='cluster',
    text='phrase',
    hover_data=['phrase', 'subreddit', 'cluster'],
    title="Interactive PCA + KMeans Clustering of Subreddit Bigrams",
    color_discrete_sequence=px.colors.qualitative.Safe  # or other color set
)

# Tweak marker and text appearance
fig.update_traces(
    textposition='top center',
    textfont_size=10,
    marker=dict(size=8, line=dict(width=0.5, color='black'))
)

# Axis labels and layout
fig.update_layout(
    xaxis_title="PCA 1",
    yaxis_title="PCA 2",
    hovermode='closest',
    legend_title="Cluster"
)

# Show the plot
fig.show(renderer='iframe')
```


<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="{{ site.baseurl }}/figures/figure_23.html"
    frameborder="0"
    allowfullscreen
></iframe>






Phrases that appear close together likely have similar meanings or occur in similar contexts in subreddit data.

Phrases that are far apart are less related in terms of usage/context.

Since TfidfVectorizer was set to ngram_range=(2, 2), it only considered bigrams. As a result, all the top terms I extracted were bigrams in PCA. 

### K-Means Clustering with Similar Word2vec embeddings


```python
# Create DataFrame for plotting
n_clusters = 6  # adjust this
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

pca_cluster_word2vec_df = pd.DataFrame({
    'x': X_reduced[:, 0],
    'y': X_reduced[:, 1],
    'phrase': valid_terms,
    'cluster': labels.astype(str)
})

# Plotly scatter plot
fig = px.scatter(
    pca_cluster_word2vec_df,
    x='x',
    y='y',
    color='cluster',
    #text='phrase',
    hover_data=['phrase'],
    title="K-Means Clustering of Phrases + Similar Terms (PCA Reduced)",
    color_discrete_sequence=px.colors.qualitative.Safe
)

fig.update_traces(
    textposition='top center',
    textfont_size=10,
    marker=dict(size=8, line=dict(width=0.5, color='black'))
)

fig.update_layout(
    xaxis_title="PCA 1",
    yaxis_title="PCA 2",
    hovermode='closest',
    legend_title="Cluster"
)

fig.show(renderer='iframe')
```

    


<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="{{ site.baseurl }}/figures/figure_25.html"
    frameborder="0"
    allowfullscreen
></iframe>



### Sentiment Analysis 


In our analysis, we examine how game metadata (e.g., mode_rank, has_coop, game_age, rating) affects the probability of anger expressed in comments of
Reddit Gaming Communities, as identified by a DistilRoBERTa-based emotion classifier.

Why [DistilRoBERTa](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)?

- Efficient yet accurate:
DistilRoBERTa is a distilled version of RoBERTa, offering about 97% of the original’s emotion classification performance while being 60% smaller and faster, crucial for processing large datasets of gaming comments efficiently.
- Emotion granularity:
The model classifies text into seven categories (6 basic emotions + neutral) based on Ekman’s framework, allowing us to focus specifically on anger probability, a key emotion linked to user frustration or dissatisfaction in game discussions.
- Neutral category importance:
Many comments contain no strong emotional content; the neutral category captures these.
Including neutral helps separate true emotional reactions (like anger) from neutral or off-topic remarks, improving the precision of modeling anger as an outcome variable.
Ignoring neutral would risk conflating lack of emotion with low anger, biasing estimates and reducing interpretability.

Why Model Anger Probability?

Instead of simple binary classification, modeling probability scores of anger lets us capture the degree of emotional intensity in comments.
This finer granularity enables us to detect subtle effects of game metadata on emotional responses both between subreddits (random effects) and within subreddit variation (fixed effects).


*For comparison/ For analyzing effects of player immersion (by perspective) on neutrality score (not implemented in this codebook):*

Using [Twitter-RoBERTa](cardiffnlp/twitter-roberta-base-sentiment-latest)

Twitter-RoBERTa is a RoBERTa-base model trained on over 124 million tweets, capturing modern social media language—including slang and informal expressions common in gaming discussions.

- Sentiment granularity:
It classifies text into three sentiment classes: Negative, Neutral, and Positive, perfectly matching the tone of gaming conversations.
Why sentiment matters:
While DistilRoBERTa captures specific emotions like anger, Twitter-RoBERTa summarizes the overall positive or negative sentiment of a comment, giving a broader picture of user attitudes toward game features.
- Neutral category importance:
The neutral class helps differentiate emotionally charged comments from neutral opinions, ensuring the model doesn’t confuse neutral feedback with positive or negative sentiment.
- Complementing emotion analysis:
Pairing Twitter-RoBERTa’s sentiment output with DistilRoBERTa’s emotion probabilities creates a richer understanding of gaming community reactions, distinguishing between general sentiment and specific emotions like anger or joy.


We label the comments in df_sampled (max.500 sampled posts per gaming subreddit, given a subreddit has at least 100 posts).

Make sure to enable GPU acceleration (e.g., Kaggle: Settings → Accelerator → GPU, Colab: Runtime → Change runtime type → GPU) and pass your models and data to the GPU to significantly speed up inference and handle large-scale emotion and sentiment classification efficiently.

```python
# Check GPU availability
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))

# Load phrases once and convert to sets for faster lookup
phrases_df = pd.read_csv('top_50_phrases_per_subreddit.csv')
phrases_per_subreddit = {
    sub: set(phrases.str.lower()) 
    for sub, phrases in phrases_df.groupby('subreddit')['phrase']
}
```

    GPU Available: True
    CUDA Device: Tesla T4
    


```python
# Initialize pipelines
pipe_distilbert = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    truncation=True,
    max_length=512,
    device=0 if torch.cuda.is_available() else -1,
    batch_size=16  # Process multiple sentences at once
)

pipe_roberta = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True,
    max_length=512,
    device=0 if torch.cuda.is_available() else -1,
    batch_size=16
)
```
## Results and Discussion


#### `Phrases and sentiment in different subreddits`




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
      <th>phrase</th>
      <th>emo_label</th>
      <th>senti_label</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Overwatch</td>
      <td>alt fire</td>
      <td>anger</td>
      <td>negative</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Overwatch</td>
      <td>alt fire</td>
      <td>neutral</td>
      <td>negative</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Overwatch</td>
      <td>alt fire</td>
      <td>neutral</td>
      <td>neutral</td>
      <td>24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Overwatch</td>
      <td>alt fire</td>
      <td>neutral</td>
      <td>positive</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Overwatch</td>
      <td>alt fire</td>
      <td>sadness</td>
      <td>negative</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3451</th>
      <td>zelda</td>
      <td>young link</td>
      <td>joy</td>
      <td>positive</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3452</th>
      <td>zelda</td>
      <td>young link</td>
      <td>neutral</td>
      <td>negative</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3453</th>
      <td>zelda</td>
      <td>young link</td>
      <td>neutral</td>
      <td>neutral</td>
      <td>26</td>
    </tr>
    <tr>
      <th>3454</th>
      <td>zelda</td>
      <td>young link</td>
      <td>neutral</td>
      <td>positive</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3455</th>
      <td>zelda</td>
      <td>young link</td>
      <td>sadness</td>
      <td>negative</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3456 rows × 5 columns</p>
</div>



To better understand the sentiment associated with each phrase, we compute aggregated statistics based on all its occurrences in the data. Since the same phrase can appear in multiple sentences—sometimes with different sentiment labels and confidence scores—we summarize the results at the phrase level.

For each (subreddit, phrase, sentiment label) combination, we calculate:

* count: How many times the phrase appeared with this sentiment label
    
* avg_score: The average model confidence score for that sentiment
    
* proportion: The fraction of times this sentiment was assigned out of all sentiment assignments for that phrase

By focusing on the proportion, we can assess which sentiment is most frequently associated with a phrase, regardless of variations in score.



#### Sentiment Distribution by Phrase 
![Sentiment Distribution by Phrase (Proportion)](figures/docana-project1_60_2.png)
Taking one of the subreddit we researched on, r/leagueoflegends, as example, "elo hell" is used in online competitive gaming (such as league of legends) to describe a situation where players feel stuck in a lower rank or division than their skill level warrants, often due to factors outside their direct control. This frustration stems from the belief that teammates, matchmaking, or other external influences hinder their ability to climb the ranks, despite individual skill. 
#### Emotion Distribution by Phrase 
![Emotion Distribution by Phrase in r/leagueoflegends](figures/docana-project1_68_5.png)

Here, again, most phrases had neutral sentiments, with only a few exceptions. Since these non-neutral phrases are limited in number. We also examined the second most dominant emotions to gain a better understanding.

#### Second Dominant Emotion
![Second Dominant Emotion](figures/docana-project1_73_0.png)



## Conclusion (p1)

The TF-IDF analysis helped identify the most distinctive phrases across different gaming communities, highlighting key terms related to gameplay, competition, and player interactions. The Word2Vec embeddings revealed meaningful semantic relationships between phrases, uncovering clusters of related terms that reflect shared themes or characteristics within each gaming community.

Our sentiment analysis of the TF-IDF phrases showed that most gaming discussions tend to be emotionally neutral rather than strongly positive or negative. While we observed some negatively charged terms, such as “people team” and “console players,” mostly from competitive multiplayer games, the limited number of phrases with clear positive or negative sentiment means we cannot draw definitive conclusions. Nonetheless, these initial findings highlight interesting emotional dynamics around game mechanics and competitiveness that motivate us to investigate these dynamics further.

Limitations: There was an imbalance in our dataset, with the majority of comments coming from League of Legends, followed by Hearthstone, while other games had significantly fewer comments. This imbalance may have affected both the sentiment analysis and the overall results. Also the fact that we looked at the sentiment of sentences that TF-IDF phrases appeared in, means we only get approximation of how phrases tend to be used or discussed but not their standalone sentiment.


## Conclusion (p2)


This analysis explored how game metadata—particularly game mode (e.g., multiplayer/cooperative play), game age, and user rating—relate to anger probability scores derived from an emotion classifier (emoBERT) applied to Reddit gaming comments. Guided by the General Learning Model (GLM) (Buckley and Anderson, 2006) and prior findings in favor of teamplay effects (Behnke et al., 2021, Smith et al., 2019), we focused on two main research questions:

- Does increased team play (higher multiplayer involvement) increase anger expression in subreddit gaming communities, and
- Does cooperative gameplay reduce such anger expression?

Using a linear mixed-effects model with random intercepts for author, subreddit, and combined genre to account for nested data and unobserved heterogeneity, we modeled fixed effects for multiplayer rank, cooperative mode, game age, and rating. Results partially supported the hypotheses. Higher multiplayer ranking was generally associated with increased anger, consistent with the first hypothesis, but not significant. However, the interaction between multiplayer rank and cooperative mode revealed that cooperative play increased anger expression at lower multiplayer levels — suggesting that cooperation may increase anger in smaller team settings. At higher multiplayer levels, the mitigating effect of cooperative team play emerged. Still, confidence interval overlap, indicating more complex social dynamics possibly influenced by other unmeasured factors.

Furthermore, our multilevel model reveladed only little variation of anger between subreddits, and even less so between genres of the associated games.

Several limitations temper these conclusions. Measurement errors in the IGDB metadata—for example, underreported cooperative features in some games—likely attenuated effect estimates. Similarly, emoBERT’s classification accuracy and the absence of contextual variables such as post timing, community feedback (upvotes/downvotes), and distinctions between online and offline play constrain explanatory power. The residual distribution’s right skew and low fixed-effects R-squared indicate that key predictors of anger remain unmodeled.

Furthermore, the Reddit gaming community represents a selective and emotionally expressive population, limiting the generalizability of findings. The comment-level approach, without user or game-level aggregation, may inflate significance due to non-independence of observations. Crucially, the lack of data distinguishing online versus offline cooperative play restricts interpretation of the cooperation effect, as these contexts likely differ substantially in social dynamics.

**In conclusion**, our findings offer preliminary support the GLM’s framework: cooperative gameplay can attenuate anger expression in online gaming discourse, especially in less complex multiplayer contexts. The data also suggests that increased multiplayer involvement tends to raise anger expression, aligning with expectations about team-based frustrations, though our findings here were not significant. Future research should refine metadata quality, incorporate richer contextual variables, and examine these dynamics across more representative gaming populations and interaction types. Despite limitations, this study demonstrates the value of integrating emotion classification with detailed game metadata to illuminate the social-emotional effects of video game play in real-world online communities.

## Contributions

| Team Member  | Contributions                                             |
|--------------|-----------------------------------------------------------|
| Alice Smith  | Data collection, preprocessing, model training, evaluation|                                                       |
| Bob Johnson  | ...                                                       |
| ...          | ...                                                       |

## References


1. Srinivasa-Desikan, Bhargav (2018). *Natural Language Processing and Computational Linguistics: A practical guide to text analysis with Python, Gensim, spaCy, and Keras*. Packt Publishing Ltd.
2. Bishop, Christopher M. (2006). *Pattern Recognition and Machine Learning (Information Science and Statistics)*. Springer-Verlag. Berlin, Heidelberg. ISBN: 0387310738.

3. Anderson, C. A., Shibuya, A., Ihori, N., Swing, E. L., Bushman, B. J., Sakamoto, A., Rothstein, H. R., & Saleem, M. (2010). Violent video game effects on aggression, empathy, and prosocial behavior in eastern and western countries: A meta-analytic review. Psychological Bulletin, 136(2), 151–173. https://doi.org/10.1037/a0018251

4. Behnke, M., Chwiłkowska, P., & Kaczmarek, L. D. (2021). What makes male gamers angry, sad, amused, and enthusiastic while playing violent video games? Entertainment Computing, 37, 100397. https://doi.org/10.1016/j.entcom.2020.100397

5. Buckley, K. E., & AndErlbaumC. A. (2006). A Theoretical Model of the Effects and Consequences of Playing Video Games. In Playing video games: Motives, responses, and consequences (pp. 363–378). Lawrence Erlbaum Associates Publishers.

6. Eron, L. D., & Huesmann, L. R. (1984). The relation of prosocial behavior to the development of aggression and psychopathology. Aggressive Behavior, 10(3), 201–211. https://doi.org/10.1002/1098-2337(1984)10:3<201::AID-AB2480100304>3.0.CO;2-S

7. Smith, M. J., Birch, P. D. J., & Bright, D. (2019). Identifying Stressors and Coping Strategies of Elite Esports Competitors. International Journal of Gaming and Computer-Mediated Simulations (IJGCMS), 11(2), 22–39. https://doi.org/10.4018/IJGCMS.2019040102