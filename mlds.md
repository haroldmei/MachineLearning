# Ads Click Prediction 

### 1. Problem Formulation
* Clarifying questions
  * What is the primary business objective of the click prediction system?
  * What types of ads are we predicting clicks for (e.g., display ads, video ads, sponsored content)?
  * Are there specific user segments or contexts we should consider (e.g., user demographics, browsing history)?
  * How will we define and measure the success of click predictions (e.g., click-through rate, conversion rate)?
  * Do we have negative feedback features (such as hide ad, block, etc)?
  * Do we have fatigue period (where ad is no longer shown to the users where there is no interest, for X days)?
  * What type of user-ad interaction data do we have access to can we use it for training our models? 
  * Do we need continual training? 
  * How do we collect negative samples? (not clicked, negative feedback). 
  
* Use case(s) and business goal
  * use case: predict which ads a user is likely to click on when presented with multiple ad options.
  * business objective: maximize ad revenue by delivering more relevant ads to users, improving click-through rates, and maximizing the value of ad inventory.
* Requirements;
    * Real-time prediction capabilities to serve ads dynamically.
    * Scalability to handle a large number of ad impressions.
    * Integration with ad serving platforms and data sources.
    * Continuous model training and updating.
* Constraints:
    * Privacy and compliance with data protection regulations.
    * Latency requirements for real-time ad serving.
    * Limited user attention, as users may quickly decide whether to click on an ad.
* Data: Sources and Availability:
    * Data sources include user interaction logs, ad content data, user profiles, and contextual information.
    * Historical click and impression data for model training and evaluation.
    * Availability of labeled data for supervised learning.
* Assumptions:
    * Users' click behavior is influenced by factors that can be learned from historical data.
    * Ad content and relevance play a significant role in click predictions.
    * The click behavior can be modeled as a classification problem.
  
* ML Formulation:
    * Ad click prediction is a ranking problem 

### 2. Metrics  
* Offline metrics 
  * CE 
  * NCE (normalized over baseline)
* Online metrics 
  * CTR (#clicks/#impressions)
  * Conversion rate (#conversion/#impression)
  * Revenue lift (increase in revenue over time)
  * Hide rate (#hidden ads/#impression)

### 3. Architectural Components  
* High level architecture 
* We can use point-wise learning to rank (LTR) 
    * The a binary classification task, where the goal is to predict whether a user will click (1) or not click (0) on a given ad impression -> given a pair of <user, ad> as input -> click or no click 
    * Features can include user demographics, ad characteristics, context (e.g., device, location), and historical behavior.
    * Machine learning models, such as logistic regression, decision trees, gradient boosting, or deep neural networks, can be used for prediction.

### 4. Data Collection and Preparation
* Data Sources
  * Users, 
  * Ads, 
  * User-ad interaction 
* ML Data types
* Labelling

### 5. Feature Engineering
* Feature selection 
  * Ads: 
    * IDs 
    * categories 
    * Image/videos
    * No of impressions / clicks (ad, adv, campaign)
  * User: 
    * ID, username
    * Demographics (Age, gender, location)
    * Context (device, time of day, etc)
    * Interaction history (e.g. user ad click rate, total clicks, etc)
  * User-Ad interaction: 
    * IDs(user, Ad), interaction type, time, location, dwell time 
* Feature representation / preparation
  * sparse features 
    * IDs: embedding layer (each ID type its own embedding layer)
  * Dense features: 
    * Engagement feats: No of clicks, impressions, etc 
    * use directly 
  * Image / Video: 
    * preprocess 
    * use e.g. SimCLR to convert -> feature vector 
  * Category: Textual data 
    * normalization, tokenization, encoding 

### 6. Model Development and Offline Evaluation
* Model selection 
  * LR 
  * Feature crossing + LR 
    * feature crossing: combine 2/more features into new feats (e.g. sum, product)
      * pros: capture nonlin interactions b/w feats 
      * cons: manual process, and domain knowledge needed 
  * GBDT 
    * pros: interpretable
    * cons: inefficient for continual training, can't train embedding layers 
  * GBDT + LR 
    * GBDT for feature selection and/or extraction, LR for classific
  * NN
    * Two options: single network, two tower network (user tower, ad tower)
    * Cons for ads prediction: 
      * sparsity of features, huge number of them 
      * hard to capture pairwise interactions (large no of them)
    * Not a good choice here. 
  * Deep and cross network (DCN)
    * finds feature interactions automatically 
    * two parallel networks: deep network (learns complex features) and cross network (learns interactions)
    * two types: stacked, and parallel 
  * Factorization Machine 
    * embedding based model, improves LR by automatically learning feature interactions (by learning embeddings for features) 
    * w0  + \sum (w_i.x_i) + \sum\sum <v_i, v_j> x_i.x_j
    * cons: can't learn higher order interactions from features unlike NN
  * Deep factorization machine (DFM)
    * combines a NN (for complex features) and a FM (for pairwise interactions)
  * start with LR to form a baseline, then experiment with DCN & DeepFM 
   
* Model Training 
  * Loss function: 
    * binary classification: CE 
    * Dataset 
      * labels: positive: user clicks the ad < t seconds after ad is shown, negative: no click within t secs  
  * Model eval and HP tuning 
  * Iterations 
  
### 7. Prediction Service
* Data Prep pipeline
  *  static features (e.g. ad img, category) -> batch feature compute (daily, weekly) -> feature store
  *  dynamic features: # of ad impressions, clicks. 
* Prediction pipeline 
  * two stage (funnel) architecture 
    * candidate generation 
      * use ad targeting criteria by advertiser (age, gender, location, etc)
    * ranking 
      * features -> model -> click prob. -> sort 
      * re-ranking: business logic (e.g. diversity)
* Continual learning pipeline 
  * fine tune on new data, eval, and deploy if improves metrics  
  
### 8. Online Testing and Deployment  
* A/B Test 
* Deployment and release 

### 9. Scaling, Monitoring, and Updates 
* Scaling (SW and ML systems)
* Monitoring 
* Updates 

### 10. Other topics  
* calibration: 
  * fine-tuning predicted probabilities to align them with actual click probabilities 
* data leakage: 
  * info from the test or eval dataset influences the training process
  * target leakage, data contamination (from test to train set)
* catastrophic forgetting 
  *  model trained on new data loses its ability to perform well on previously learned tasks 




# Design an event recommendation system 

## 1. Problem Formulation 

* Clarifying questions 
  - Use case? 
    - event recommendation system similar to eventbrite's. 
  - What is the main Business objective? 
    - Increase ticket sales  
  - Does it need to be personalized for the user? Personalized for the user
  - User locations? Worldwide (multiple languages) 
  - User’s age group: 
  - How many users? 100 million DAU
  - How many events? 1M events / month 
  - Latency requirements  - 200msec?
  - Data access 
    - Do we log and have access to any data? Can we build a dataset using user interactions ?
    - Do we have textual description of items? 
    - Can we use location data (e.g. 3rd party API)? (events are location based)
  - Can users become friends on the platform? Do we wanna use friendships?
  - Can users invite friends? 
  - Can users RSVP or just register?
  - Free or Paid? Both 

* ML formulation 
  * ML Objective: Recommend most relevant (define) events to the users to maximize the number of registered events
  * ML category: Recommendation system (ranking approach)
    * rule based system 
    * embedding based (CF and content based)
    * Ranking problem (LTR)
      * pointwise, pairwise, listwise 
    * we choose pointwise LTR ranking formulation 
  * I/O: In: user_id, Out: ranked list of events + relevance score
    * Pointwise LTR classifier I/O: I: <user_id, event_id>, O: P(event register) (Binary classification)

## 2. Metrics (Offline and Online) 

* Offline: 
    * precision @k, recall @ k (not consider ranking quality)
    * MRR, mAP, nDCG (good, focus on first element, binary relevance, non-binary relevance) -> here event register binary relevance so use mAP  
   
* Online: 
    * CTR, conversion rate, bookmark/like rate, revenue lift  

## 3. Architectural Components (MVP Logic) 
* We two stage (funnel) architecture for 
  * candidate generation 
    * rule based event filtering (e.g. location, etc)
  * ranking formulation (pointwise LTR) binary classifier  

## 4. Data preparation 

* Data Sources: 
  1. Users (user profile, historical interactions)
  2. Events 
  3. User friendships 
  4. User-event interactions
  5. Context


*  Labeling: 

## 5. Feature engineering 

* Note: Event based recommendation is more challenging than movie/video: 
   * events are short lived -> not many historical interactions -> cold start (constant new item problem)
   * So we put more effort on feature engineering (many meaningful features)

* Features: 
  - User features 
    - age (one hot), gender (bucketize), event history  
 
  - Event features 
    - price, No of registered, 
    - time (event time, length, remained time)
    - location  (city, country, accessibility)
    - description
    - host (& popularity)
  
  - User Event features 
    - event price similarity 
    - event description similarity 
    - no. registered similarity 
    - same city, state, country
    - distance 
    - time similarity (event length, day, time of day)
  
  - Social features 
    - No./ ratio of friends going 
    - invited by friends (No)
    - hosted by friend (similarity)
  
  - context 
    - location, time  

* Feature preprocessing 
  - one hot (gender)
  - bucketize + one hot (age, distance, time)

* feature processing 
  * Batch (for static) vs Online (streaming, for dynamic) processing 
  * efficient feature computation (e.g. for location, distance)
  * improve: embedding learning - for users and events 

## 6. Model Development and Offline Evaluation 

* Model selection 
  * Binary classification problem: 
    * LR (nonlinear interactions)
    * GBDT (good for structured, not for continual learning)
    * NN (continual learning, expressive, nonlinear rels)
  * we can start with GBDT as a baseline and experiment improvements by NN (both good options)
* Dataset 
  * for each user and event pair, compute features, and label 1 if registered, 0 if not 
  * class imbalance 
    * resampling 
    * use focal loss or class-balanced loss 

## 7. Prediction Service 
* Candidate generation 
  * event filtering (millions to hundreds)
    * rule based (given a user, e.g. location, type, etc filters)
* Ranking 
  * compute scores for <usr, event> pairs, and sort 

## 8. Online Testing and Deployment  
Standard approaches as before.  

## 9. Scaling




# Design a game recommendation engine 

## 1. Problem Formulation 
User-game interaction 

Some existing data examples:  
* Games data

  * app_id,
   title,
   date_release,
   win,
   mac,
   linux,
   rating,
   positive_ratio,
   user_reviews,
   price_final,
   price_original,
   discount,
   steam_deck,

* User historic data 
   
  *  user_id,
   products,
   reviews,


* Recommendations data 
  
  * app_id,
  helpful,
  funny,
  date,
  is_recommended,
  hours,
  user_id,
  review_id,
  
* Reviews 


* Example Open Source Data: [Steam games complete dataset](https://www.kaggle.com/datasets/trolukovich/steam-games-complete-dataset) ([CF and content based github](https://github.com/AudreyGermain/Game-Recommendation-System))
  * Game fatures include:  
Url, 
types
name,
desc_snippet,
recent_reviews,
all_reviews,
release_date,
developer,
publisher,
popular_tag,

### Clarifying questions 
- Use case? Homepage?
  - Does user sends a text query as well?
- Business objective? 
  - Increase user engagement (play, like, click, share), purchase?, create a better ultimate gaming experience 
- Similar to previously played, or personalized for the user? Personalized for the user
- User locations? Worldwide (multiple languages) 
- User’s age group: 
- Do users have any favorite lists, play later, etc?
- How many games? 100 million
- How many users? 100 million DAU
- Latency requirements  - 200msec?
- Data access 
  - Do we log and have access to any data? Can we build a dataset using user interactions ?
  - Do we have textual description of items? 
- can users become friends on the platform and do we wanna take that into account?
- Free or Paid?  




### ML objective

- Recommend most engaging (define) games
  * Max. No. of clicks (clickbait)
  * Max. No. completed games/sessions/levels (bias to shorter)
  * Max. total hours played ()
  * Max. No. of relevant items (proxy by user implicit/explicit reactions) -> more control over signals, not the above shortcomings

* Define relevance: e.g. like is relevant, or playing half of it is, …
* ML Objective: build dataset and model to predict the relevance score b/w user and a game
* I/O: I: user_id, O: ranked list of games + relevance score
* ML category: Recommendation System

## 2. Metrics (Offline and Online) 

* Offline: 
    * precision @k, mAP, and diversity 
* Online: 
    * CTR, # of completed, # of purchased, total play time, total purchase, user feedback 

## 3. Architectural Components (MVP Logic) 
The main approaches used for personalized recommendation systems: 
* Content-based filtering: suggest items similar to those user found relevant (e.g. liked)
    * No need for interaction data, recommends new items to users (no item cold start)
    * Capture unique interests of users
    * New user cold start 
    * Needs domain knowledge 
* CF: Using user-user (user based CF) or item-item similarities (item based CF)
    * Pros
        * No domain knowledge 
        * Capture new areas of interest 
        * Faster than content (no content info needed)
    * Cons: 
        * Cold start problem (both user and item)
        * No niche interest 
* Hybrid 
    * Parallel hybrid: combine(CF results, content based)
    * Sequential: [CF based] -> Content based

What do we choose? 
We choose a sequential hybrid model (standard e.g. for video recommendation)

We follow  the three stage recommender system (funnel architecture) in order to meet latency requirements and eb able to scale the system to billions of items. 

```mermaid
   Candidate generation --> Ranking --> Re-ranking
```

In the first stage, we use a light model to retrive thousands of items from millions
In the second (ranking) stage, we focus on high precision using a powerful model. This will not impact serving speed much because it's only run on smaller subset of items. 

Candidate generation in practice comes from aggregation of different candidate generation models. Here we can assume three candidate generation modules: 

1. Candidate generation 1 (Relevance based)
2. Candidate generation 2 (Popularity)
3. Candidate generation 3 (Trending)  

where we use CF for candidate generation 1

We use content based modeling for ranking.

## 4. Data preparation 

Data Sources: 

1. Users (user profile, historical interactions):
     * User profile
       * User_id, username, age, gender, location (city, country), lang, timezone


2. Games (structures, metadata, game content - what is it?)
   - Game_id, title, date, rating, expected_length?,  #reviews, language, tags, description, price, developer, publisher, level, #levels

3. User-Game interactions:  
Historical interactions: Play, purchase, like, and search history, etc  
   - User_id, game_id, timestamp, interaction_type(purchase, play, like, impression, search), interaction_val, location


1. Context: time of the day, day of the week, device, OS

Type

- Removing duplicates 
- filling missing values 
- normalizing data.

### Labeling: 
For features in the form of <user, video> pairs -> labeling strategy based on explicit or implicit feedback 
e.g. "positive" if user liked the item explicitly or interacted (e.g. watched/played) at least for X (e.g. half of it).   
negative samples: sample from background distribution -> correct via importance smapling 

## 5. Feature engineering 

There are several machine learning features that can be extracted from games. Here are some examples:

- Game metadata features
- Game state: e.g. the positions of players, the status of objects and obstacles, the time remaining, and the score.
- Game mechanics: The rules and interactions that govern the game. 
- User engagement: e.g. the length of play sessions, frequency of play, and player retention rates.
- Social interactions: b/w players: to identify patterns of behavior, such as the formation of alliances, the sharing of resources, and the types of communication used between players.
- Player preferences: which game features are most popular among players, which can help inform game design decisions.
- Player behaviors: player movement patterns, the types of actions taken by players, and the strategies used to achieve objectives.


We select some important features as follows:

* Game metadata features: 
  * Game ID, 
  Duration, 
  Language, 
  Title, 
  Description,
  Genre/Category, 
  Tags,  
  Publisher(popularity, reviews), 
  Release date, 
  Ratings, 
  Reviews, 
  (Game content ?)
game titles, genres, platforms, release dates, user ratings, and user reviews.



* User profile: 
  * User ID, Age, Gender, Language, City, Country 

* User-item historical features: 
  * User-item interactions 
    * Played, liked, impressions
    * purchase history (avg. price)
  * User search history 

* Context


### Feature representation: 

* Categorical data (game_id, user_id, language, city): Use embedding layers, learned during
training 
* Categorical_data(gender, age): one_hot
* Continuous variables: normalize, or bucketize and one-hot (e.g. price) 
* Text:(title, desc, tags): title/description use embeddings, pre-trained BERT, fine tune on game language?, tags: CBOW
* 
* Game content embeddings? 

## 6. Model Development and Offline Evaluation 

### 6.1 Candidate Generation 

For candidate generation 1 (Relevance Based), we choose CF. 

For CF there are two embedding based modeling options: 
1. Matrix Factorization 
   * Pros: Training speed (only two matrices to learn), Serving speed (static learned embeddings)
   * Cons: only relies on user-item interactions (No user profile info e.g. language is used); new-user cold start problem 
2. Two tower neural network:
   * Pros: Accepts user features (user profile + user search history) -> better quality recommendation; handles new users 
   * Cons: Expensive training, serving speed     

We chose two-tower network here. 

#### Two-tower network
  * two encoder towers (user tower + encoder tower)
  * user tower encodes user features into user embeddings  $u$ 
  * item tower encodes item features into item embeddings   $v_i$
  * similarity $u$, $v_i$ is considered as a relevance score (ranking as classification problem)


#### Loss function: 
Minimize cross entropy for each positive label and sampled negative examples 

### 6.2 Ranking 
For Ranking stage, we prioritize precision over efficiency. We choose content based filtering. Choose a model that relies in item features.  
ML Obj options: 
   - max P(watch| U, C)
   - max expected total watch time 
   - multi-objective (multi-task learning: add corresponding losses)
  
Model Options: 
- FF NN (e.g. similar tower network to a tower network) + logistic regression 
- Deep Cross Network (DCN)

Features  

* Video ID embeddings (watched video embedding avg, impression video embedding), 
* Video historic
  *  No. of previous impressions, reviews, likes, etc
  *  Time features (e.g. time since last play), 
* Language embedding (user, item), 
* User profile 
* User Historic (e.g. search history)



### 6.3 Re-Ranking 
Re-ranks items by additional business criteria (filter, promote)  
We can use ML models for clickbait, harmful content, etc or use heuristics   
Examples: 
* Age restriction filter 
* Region restriction filter 
* Video freshness (promote fresh content)
* Deduplication 
* Fairness, bias, etc 




## 7. Prediction Service 
two-tower network inference: find the k-top most relevant items given a user ->  
It's a classic nearest neighbor problem -> use approximate nearest neighbor (ANN) algorithms  

## 8. Online Testing and Deployment  
Standard approaches as before.  
## 9. Scaling
The three stage candidate generation - ranking - re-ranking can be scaled well as described earlier. It also meets the requirements of speed (funnel architecture), precision(ranking component), and diversity (multiple candid generation). 

### Cold start problem: 
* new users: two tower architectures accepts new users and we can still use user profile info even with no interaction 
* new items: recommend to random users and collect some data - then fine tune the model using new data

### Training: 
We need to be able to fine tune the model 
### Exploration exploitation trade-off 
- Multi-armed bandit (an agent repeatedly selects an option and receives a reward/cost. The goal of to maximize its cumulative reward over time, while simultaneously learning which options are most valuable.)
### Other Extensions: 
* [Multi-task learning](https://daiwk.github.io/assets/youtube-multitask.pdf)
  * Includes a shared feature extractor that is trained jointly with multiple prediction heads, each of which is responsible for predicting a different aspect of user behavior, such as click-through rate, watch time, and view count. The model is trained using a combination of supervised and unsupervised learning techniques, including cross-entropy loss, pairwise ranking loss, and self-supervised contrastive learning.
* Positional bias (detection and correction) 
* Selection bias (detection and correction)
* Add negative feedback (dislike)
* Locality preservation: 
  * Use sequential user behavior info (CBOW model)
* effect of seasonality 
* what if we only have a query and personal (item, provider) history? 
  * item embeddings, provider embeddings, query embeddings 
  * we can build a query-aware attention mechanism that computes 

### More resources 

* [Content-based](https://www.kaggle.com/code/fetenbasak/content-based-recommendation-game-recommender), [NLP analysis](https://www.kaggle.com/code/greentearus/steam-reviews-nlp-analysis), [Collaborative Denoising AE](https://www.kaggle.com/code/krsnewwave/collaborative-denoising-autoencoder-steam)  
* [User-based CF, item-based CF and MF](https://github.com/manandesai/game-recommendation-engine) ([github](https://github.com/manandesai/game-recommendation-engine/blob/main/recommenders.ipynb))
* [CF and content based](https://github.com/AudreyGermain/Game-Recommendation-System) 


# Harmful content detection on social media

### 1. Problem Formulation
* Clarifying questions
  * What types of harmful content are we aiming to detect? (e.g., hate speech, explicit images, cyberbullying)?
  * What are the potential sources of harmful content? (e.g., social media, user-generated content platforms)
  * Are there specific legal or ethical considerations for content moderation
  * What is the expected volume of content to be analyzed daily?
  * What are supported languages? 
  * Are there human annotators available for labeling? 
  * Is there a feature for users to report harmful content? (click, text, etc). 
  * Is explainablity important here? 
  
* Integrity deals with: 
    * Harmful content (focus here)
    * Harmful act/actors  
* Goal: monitor posts, detect harmful content, and demote/remove 
* Examples harmful content categories: violence, nudity, hate speech 
* ML objective: predict if a post is harmful 
  * Input: Post (MM: text, image, video) 
  * Output:  P(harmful) or P(violent), P(nude), P(hate), etc
* ML Category: Multimodal (Multi-label) classification 
* Data: 500M posts / day (about 10K annotated)
* Latency: can vary for different categories 
* Able to explain the reason to the users (category) 
* support different languages? Yes 

### 2. Metrics  
- Offline 
  - F1 score, PR-AUC, ROC-AUC 
- Online 
  - prevalence (percentage of harmful posts didn't prevent over all posts), harmful impressions, percentage of valid (reversed) appeals, proactive rate (ratio of system detected over system + user detected) 

### 3. Architectural Components  
* Multimodal input (text, image, video, etc): 
  * Multimodal fusion techniques 
    * Early Fusion: modalities combined first, then make a single prediction 
    * Late Fusion: process modalities independently, fuse predictions
      * cons: separate training data for modalities, comb of individually safe content might be harmful 
* Multi-Label/Multi-Task classification 
  * Single binary classifier (P(harmful))
    * easy, not explainable 
  * One binary classifier per harm category (p(violence), p(nude), p(hate))
    * multiple models, trained and maintained separately, expensive 
  * Single multi-label classifier 
    * complicated task to learn 
  * Multi-task classifier: learn multi tasks simultanously 
    * single shared layers (learns similarities between tasks) -> transformed features 
    * task specific layers: classification heads 
    * pros: single model, shared layers prevent redundancy, train data for each task can be used for others as well (limited data)

### 4. Data Collection and Preparation

* Main actors for which data is available: 
  * Users 
    * user_id, age, gender, location, contact
  * Items(Posts) 
    * post_id, author_id, text context, images, videos, links, timestamp
  * User-post interactions 
    * user_id, post_id, interaction_type, value, timestamp


### 5. Feature Engineering
Features: 
Post Content (text, image, video) + Post Interactions (text + structured) + Author info + Context  
* Posts 
  * Text:  
    * Preprocessing (normalization + tokenization) 
    * Encoding (Vectorization): 
      * Statistical (BoW, TF-IDF)
      * ML based encoders (BERT)
    * We chose pre-trained ML based encoders (need semantics of the text)
    * We chose Multilingual Distilled (smaller, faster) version of BERT (need context), DistilmBERT 
  * Images/ Videos:   
    * Preprocessing: decoding, resize, scaling, normalization
    * Feature extraction: pre-trained feature extractors 
      * Images: 
        * CLIP's visual encoder 
        * SImCLR 
      * Videos: 
        * VideoMoCo
* Post interactions: 
  * No. of likes, comments, shares, reports (scale) 
  * Comments (text): similar to the post text (aggregate embeddings over comments)
* Users: 
  * Only use post author's info
    * demographics (age, gender, location)
    * account features (No. of followers /following, account age)
    * violation history (No of violations, No of user reports, profane words rate)
* Context: 
  * Time of day, device

### 6. Model Development and Offline Evaluation
* Model selection 
  * NN: we use NN as it's commonly used for multi-task learning 
* HP tuniing: 
  * No of hidden layers, neurons in layers, act. fcns, learning rate, etc
  * grid search commonly used 
* Dataset: 
  * Natural labeling (user reports) - speed 
  * Hand labeling (human contractors) - accuracy 
  * we use natural labeling for train set (speed) and manual for eval set (accuracy)
* loss function: 
  * L = L1 + L2 + L3 ... for each task 
  * each task is a binary classific so e.g. CE for each task  
* Challenge for MM training: 
  * overfitting (when one modality e.g. image dominates training)
    * gradient blending and focal loss 

### 7. Prediction Service
* 3 main components: 
  * Harmful content detection service 
  * Demoting service (prob of harm with low confidence)
  * violation service (prob of harm with high confidence)

### 8. Online Testing and Deployment  

### 9. Scaling, Monitoring, and Updates

### 10. Other topics 
* biases by human labeling 
* use temporal information (e.g. sequence of actions)
* detect fake accounts 
* architecture improvement: linear transformers 


# Image Search System (Pinterest)

### 1. Problem Formulation
* Clarifying questions
    - What is the primary (business) objective of the visual search system?
    - What are the specific use cases and scenarios where it will be applied?
    - What are the system requirements (such as response time, accuracy, scalability, and integration with existing systems or platforms)?
    - How will users interact with the system? (click, like, share, etc)? Click only
    - What types of visual content will the system search through (images, videos, etc.)? Images only 
    - Are there any specific industries or domains where this system will be deployed (e.g., fashion, e-commerce, art, industrial inspection)?
    - What is the expected scale of the system in terms of data and user interactions?
    - Personalized? not required 
    - Can we use metadata? In general yes, here let's not. 
    - Can we assume the platform provides images which are safe? Yes
* Use case(s) and business goal
  * Use case: allowing users to search for visually similar items, given a query image by the user 
  * business goal: enhance user experience, increase click through rate, conversion rates, etc (depends on use case)
* Requirements
  * response time, accuracy, scalability (billions of images)
* Constraints
  * budget limitations, hardware limitations, or legal and privacy constraints
* Data: sources and availability
  * sources of visual data: user-generated, product catalogs, or public image databases?
  * Available? 
* Assumptions
* ML formulation: 
  * ML Objective: retrieve images that are similar to query image in terms of visual content 
  * ML I/O: I: a query image, and O: a ranked list of most similar images to the query image 
  * ML category: Ranking problem (rank a collection of items based on their relevance to a query)

### 2. Metrics  
* Offline metrics 
  * MRR 
  * Recall@k 
  * Precision@k 
  * mAP 
  * nDCG 
* Online metrics 
  * CTR 
  * Time spent on images 

### 3. Architectural Components  
* High level architecture 
  * Representation learning: 
    * transform input data into representations (embeddings) - similar images are close in their embedding space 
    * use distance between embeddings as a similarity measure between images 

### 4. Data Collection and Preparation
* Data Sources
  * User profile
  * Images 
    * image file
    * metadata
  *  User-image interactions: impressions, clicks: 
  * Context 
* Data storage
* ML Data types
* Labelling

### 5. Feature Engineering
* Feature selection 
  * User profile : User_id, username, age, gender, location (city, country), lang, timezone
  * Image metadata: ID, user ID, tags, upload date, ... 
  * User-image interactions: impressions, clicks: 
    * user id, Query img id, returned img id, interaction type (click, impression), time, location
* Feature representation 
  * Representation learning (embedding)
* Feature preprocessing 
  * common feature preprocessing for images: 
    * Resize (e.g. 224x224), Scale (0-1), normalize (mean 0, var 1), color mode (RGB, CMYK) 

### 6. Model Development and Offline Evaluation
* Model selection 
  * we choose NN because of 
    * unstructured data (images, text) -> NN good at it 
    * embeddings needed 
  * Architecture type: 
    * CNN based e.g. ResNet 
    * Transformer based (ViT)
    * Example: Image -> Convolutional layers -> FC layers -> embedding vector  
* Model Training 
  * contrastive learning -> used for image representation learning 
    * train to distinguish similar and dissimilar items (images)
* Dataset 
  * each data point: query img, positive sample (similar to q), n - 1 neg samples (dissimilar)
    * query img : randomly choose 
    * neg samples: randomly choose 
    * positive samples: human judge, interactions (e.g. click) as a proxy, artificial image generated from q (self supervision)
      * human: expensive, time consuming 
      * interactions: noisy and sparse 
      * artificial: augment (e.g. rotate) and use as a positive sample (similar to simCLR or MoCo) - data distribution differs in reality 
* Loss Function: contrastive loss 
  * contrastive loss: 
    * works on pairs (Eq, Ei)
    * calculate distance: b/w pairs -> softmax -> cross entropy <- Labels 
* Model eval and HP tuning 
* Iterations 
  
### 7. Prediction Service
* Prediction pipeline 

  * Embedding generation service 
    * image -> preprocess -> embedding gen (ML model) -> img embedding 
  * NN search service 
    * retrieve the most similar images from embedding space 
      * Exact: O(N.D)
      * Approximate(ANN) - sublinear e.g. O(D.logN)
        * Tree based ANN (e.g. R-trees, Kd-trees) 
          * partition space into two (or more) at each non-leaf node, 
          * only search the partition for query q 
        * Locality Sensitive Hashing LSH 
          * using hash functions to group points into buckets (close points into same buckets)
        * Clustering based 
    * We use ANN using an existing library like Faiss (Facebook)
  * Re-ranking service 
    * business level logic and policies (e.g. filter inappropriate or private items, deduplicate, etc)
* Indexing pipeline
  * Indexing service: indexes images by their embeddings 
  * keep the table updated for new images 
  * increases memory usage -> use optimization (vector / product quantization)

### 8. Online Testing and Deployment  
* A/B Test 
* Deployment and release 

### 9. Scaling, Monitoring, and Updates 
* Scaling (SW and ML systems)
* Monitoring 
* Updates 

### 10. Other points: 



# Multimodal Video Search System 

### 1. Problem Formulation
* Clarifying questions
    - What is the primary (business) objective of the search system?
    - What are the specific use cases and scenarios where it will be applied?
    - What are the system requirements (such as response time, accuracy, scalability, and integration with existing systems or platforms)?
    - What is the expected scale of the system in terms of data and user interactions?
    - Is their any data available? What format? 
    - Can we use video metadata? Yes 
    - Personalized? not required 
    - How many languages needs to be supported?
    
* Use case(s) and business goal
  * Use case: user enters text query into search box, system shows the most relevant videos 
  * business goal: increase click through rate, watch time, etc.  
* Requirements
  * response time, accuracy, scalability (50M DAU)
* Constraints
  * budget limitations, hardware limitations, or legal and privacy constraints
* Data: sources and availability
  * Sources: videos (1B), text 
  * 10M pairs of <video, text_query>. Videos have metadata (title, description, tags) in text format 
* Assumptions
* ML formulation: 
  * ML Objective: retrieve videos that are relevant to a text query  
  * ML I/O: I: text query from a user, O: ranked list of relevant videos on a video sharing platform  
  * ML category: Visual search + Text Search systems 

   
### 2. Metrics  
- Offline
  - Precision@k, mAP, Recall@k, MRR 
  - we choose MRR (avg rank of first relevant element in results) due to the format of our eval data <video, text> pair 
- Online 
  - CTR: problem: doesn't track relevancy, click baits  
  - video completion rate: partially watched videos might still found relevant by user 
  - total watch time
  - we choose total watch time: good indicator of relevance 

### 3. Architectural Components  
Multimodal search (video, text) for video content from text query: 
- Visual search system 
  - Text query -> videos (based on similarity of text and visual content) 
  - Two tower embedding architecture (video and text_query encoders)
- Textual search system 
  - search for most similar titles, descs, and tags  w/ text query 
  - we can use Inverted Index (e.g. elastic search) for efficient full text search 
    - An inverted index is a data structure that maps terms (words) to the documents or locations where they appear, enabling efficient text-based document retrieval, commonly used in search engines.

### 4. Data Collection and Preparation
We use provided annotated data in the format of <video_id, query>. 
### 5. Feature Engineering
- Preprocessing unstructured data 
  - Text pre-processing : normalization, tokenization, token to ids
  - Video preprocessing: decode into frames -> sample -> resize -> scale, normalize, color correct 

### 6. Model Development and Offline Evaluation
* Model Selection  
  - Text encoders: 
    - Text -> Vector (Embeddings)  
    - Two approaches: 
      - Statistical (BoW, TF-IDF)
      - ML encoders (word2vec, transformer based e.g. BERT)  
    - We chose transformer based (BERT). 

  - Video encoders: 
    - Video-level
      - more expensive, but captures temporal understanding
      - Example: ViViT (Video Vision Transformer)
    - Frame-level (from sample frames and aggregate)
      - less expensive (training and serving speed, compute power) 
      - Example: ViT 


* Model Training   
  - contrastive learning (similar to visual search system). 

### 7. Prediction Service
Components: 
- Visual search from text query 
  - text -> preprocess -> encoder -> embedding 
  - videos are indexed by their encoded embeddings 
  - search: using approximate nearest neighbor search (ANN)
- Textual search
  - using Elasticsearch (full text / fuzzy search)
- Fusion  
  - re-rank based on weighted sum of rel scores 
  - re-rank using a model 
- Re-ranking 
  - business level logic and policies 

### 8. Online Testing and Deployment  

### 9. Scaling, Monitoring, and Updates


# News Feed System 

### 1. Problem Formulation
show feed (recent posts and activities from other users) on a social network platform 
* Clarifying questions
  * What is the primary business objective of the system? (increase user engagement)
  * Do we show only posts or also activities from other users?
  * What types of engagement are available? (like, click, share, comment, hide, etc)? Which ones are we optimizing for? 
  * Do we display ads as well? 
  * What types of data do the posts include? (text, image, video)?
  * Are there specific user segments or contexts we should consider (e.g., user demographics)?
  * Do we have negative feedback features (such as hide ad, block, etc)?
  * What type of user-ad interaction data do we have access to can we use it for training our models? 
  * Do we need continual training? 
  * How do we collect negative samples? (not clicked, negative feedback). 
  * How fast the system needs to be? 
  * What is the scale of the system? 
  * Is personalization needed? Yes 
  
* Use case(s) and business goal
  * use case: show friends most engaging (and unseen) posts and activities on a social network platform app (personalized to user)
  * business objective: Maximize user engagement (as a set of interactions)

* Requirements;
    * Latency: 200 msec of newsfeed refreshed results after user opens/refreshes the app
    * Scalability: 5 B total users, 2 B DAU, refresh app twice 
    
* Constraints:
    * Privacy and compliance with data protection regulations.
    
* Data: Sources and Availability:
    * Data sources include user interaction logs, ad content data, user profiles, and contextual information.
    * Historical click and impression data for model training and evaluation.

* Assumptions:
    * Users' engagement behavior can be characterized by their explicit (e.g. like, click, share, comment, etc) or implicit interactions (e.g. dwell time) 
  
* ML Formulation:
    * Objective: 
      * maximize number of explicit, implicit, or both type of reactions (weighted)
      * implicit: more data, explicit: stronger signal, but less data -> weighted score of different interactions: share > comment > like > click etc 
    * I/O: I: user_id, O: ranked list of unseen posts sorted by engagement score (wighted sum) 
    * Category: Ranking problem: can be solved as pointwise LTR with multi/label (multi-task) classification

### 2. Metrics  
* Offline 
  * ROC AUC (trade-off b/w TPR and FPR)
* Online 
  * CTR, 
  * Reactions rate (like rate, comment rate, etc)
  * Time spent 
  * User satisfaction (survey)

### 3. Architectural Components  
* High level architecture 
  * We can use point-wise learning to rank (LTR) formulation 
  * Options for multi-label/task classification: 
    * Use N independent classifiers (expensive to train and maintain) 
    * Use a multi-task classifier
      * learn multi tasks simultaneously 
      * single shared layers (learns similarities between tasks) -> transformed features 
      * task specific layers: classification heads 
      * pros: single model, shared layers prevent redundancy, train data for each task can be used for others as well (limited data)

### 4. Data Collection and Preparation
* Data Sources
  * Users, 
  * Posts, 
  * User-post interaction 
  * User-user (friendship)

* Labelling

### 5. Feature Engineering

* Feature selection 
  * Posts: 
    * Text
    * Image/videos
    * No of reactions (likes, shares, replies, etc)
    * Age 
    * Hashtags 
  * User: 
    * ID, username
    * Demographics (Age, gender, location)
    * Context (device, time of day, etc)
    * Interaction history (e.g. user click rate, total clicks, likes, et )
  * User-Post interaction: 
    * IDs(user, Ad), interaction type, time, location 
  * User-user(post author) affinities 
    * connection type 
    * reaction history (No liked/commented/etc posts from author)

* Feature representation / preparation
  * Text: 
    * use a pre-trained LM to get embeddings
    * use BERT here (posts are in phrases usually, context aware helps) 
  
  * Image / Video: 
    * preprocess 
    * use pre-trained models e.g. SimCLR / CLIP to convert -> feature vector 
  
  * Dense numerical features: 
    * Engagement feats (No of clicks, etc)
      * use directly + scale the range
  * Discrete numerical: 
    * Age: bucketize into categorical then one hot 
  * Hashtags: 
    *  tokenize, token to ID, simple vectorization (TF-IDF or word2vec) - no context 


### 6. Model Development and Offline Evaluation

* Model selection 
  * We choose NN 
    * unstructured data (text, img, video)
    * embedding layers for categorical features
    * fine tune pre-trained models used for feat eng.
  * multi-labels 
    * P(click), P(like), P(Share), P(comment)
  * Two options: 
    * N NN classifiers  
    * Multi task NN (choose this) 
      * Shared layers 
      * Classification heads (click, like, share, comment)
  * Passive users problem: 
    * All their Ps will be small 
    * add two more heads 
      * Dwell time (seconds spent on post)
      * P(skip) (skip = spend time < t)
      

* Model Training 
  * Loss function: 
    * L = sum of L_is for each task 
    * for binary classif tasks: CE 
    * for regression task: MAE, MSE, or Huber loss
  * Dataset 
    * use features, post features, interactions, labels
    * labels: positive, negative for each task (like, didn't like etc)
      * for dwell time: it's a regression 
    * Imbalanced dataset: downsample negative 
  * Model eval and HP tuning 
  * Iterations 
  
### 7. Prediction Service
* Data Prep pipeline
  *  static features -> batch feature compute (daily, weekly) -> feature store
  *  dynamic features: # of post clicks, etc _> streaming  

* Prediction pipeline 
  * two stage (funnel) architecture 
    * candidate generation / retrieval service 
      * rule based 
      * filter and fetch unseen posts by users under certain criteria 
    * Ranking 
      * features -> model -> engagement prob. -> sort 
      * re-ranking: business logic, additional logic and filters (e.g. user interest category)
* Continual learning pipeline 
  * fine tune on new data, eval, and deploy if improves metrics  
  
### 8. Online Testing and Deployment  
* A/B Test 
* Deployment and release 

### 9. Scaling, Monitoring, and Updates 
* Scaling (SW and ML systems)
* Monitoring 
* Updates 

### 10. Other topics  
* Viral posts / Celebrities posts
* New users (cold start)
* Positional data bias 
* Update frequency 
* calibration: 
  * fine-tuning predicted probabilities to align them with actual click probabilities 
* data leakage: 
  * info from the test or eval dataset influences the training process
  * target leakage, data contamination (from test to train set)
* catastrophic forgetting 
  *  model trained on new data loses its ability to perform well on previously learned tasks 


# Friends/Follower recommendation (People you may know)

### 1. Problem Formulation
Recommend a list of users that you may want to connect with 
* Clarifying questions
  * What is the primary business objective of the system? 
  * What's the primary use case of the system?
  * Are there specific factors needs to be considered for recommendations?
  * Are friendships/connections symmetrical?
  * What is the scale of the system? (users, connections)
  * can we assume the social graph is not very dynamic?
  * Do we need continual training? 
  * How do we collect negative samples? (not clicked, negative feedback). 
  * How fast the system needs to be? 
  * Is personalization needed? Yes 
  
## 
* Use case(s) and business goal
  * use case: PYMMK: recommend a list of users to connect with on social media app (e.g. facebook, linkedin)
  * business objective: maximize number of formed connections 

* Requirements;
    * Scalability: 1 B total users, on avg. 10000 connection per user     
  
* Constraints:
    * Privacy and compliance with data protection regulations.
    
* Data: Sources and Availability:

* Assumptions:
    * symmetric firendships
  
* ML Formulation:
    * Objective: 
      * maximize number of formed connections 
    * I/O: I: user_id, O: ranked list of recommended users sorted by the relevance to the user 
    * ML Category: two options: 
      * Ranking problem: 
        * pointwise LTR - binary classifier (user_i, user_j) -> p(connection)
        * cons: doesn't capture social connections 
      * Graph representation (edge prediction)
        * supplement with graph info (nodes, edges)
        * input: social graph, predict edge b/w nodes 

### 2. Metrics  
* Offline 
  * GNN model: binary classification -> ROC-AUC 
  * Recommendation system: binary relationships -> mAP 
  
* Online 
  * No of friend requests sent over X time 
  * No of friend requests accepted over X time 
  
### 3. Architectural Components  
* High level architecture 
  * Node-level predictions 
  * Edge-level predictions 
  
### 4. Data Collection and Preparation
* Data Sources
  * Users, 
    * demographics, edu and work backgrounds, skills, etc
    * note: standardized data (e.g. cs / computer science)
  * User-user connections,  
  * User-user interactions, 

* Labelling

### 5. Feature Engineering

* Feature selection
    
  * User: 
    * ID, username
    * Demographics (Age, gender, location)
    * Account/Network info: No of connections, followers, following, requests, etc, account age
    * Interaction history (No of likes, shares, comments)
    * Context (device, time of day, etc)
    
  * User-user connections: 
    * Connection: IDs(user1, user2), connection type, timestamp, location 
    * edu and work affinity: major similarity, companies in common, industry similarity, etc 
    * social affinity: No. mutual connections (time discounted)
  * User-user interactions:  
    * IDs(u user1, user2), interaction type, timestamp 




### 6. Model Development and Offline Evaluation

* Model selection 
  * We choose GNN 
    * operate on graph data 
    * predict prob of edge 
    * input: graph (node and edge features)
    * output: embedding of each node
    * use similarities b/w node embeddings for edge prediction 


* Model Training 
  * snapshot of G at t. model predict connections at t+1
  * Dataset 
    * create a snapshot at time t
    * compute node and edge features 
    * create labels using snapshot at t + 1 (if connection formed, positive) 
  * Model eval and HP tuning 
  * Iterations 
  
### 7. Prediction Service
* Prediction pipeline 
  * Candidate generation 
    * Friends of Friends (FoF) - rule based - from 1B to 1K.1K = 1M candidates -> FoF service  
  * Scoring service (using GNN model -> embeddings -> similarity scores)
  * sort by score 
* pre-compute PYMK tables for each / active users and store in DB 
* re-rank based on business logic 
  
### 8. Online Testing and Deployment  
* A/B Test 
* Deployment and release 

### 9. Scaling, Monitoring, and Updates 
* Scaling (SW and ML systems)
* Monitoring 
* Updates 

### 10. Other topics  
* add a lightweight ranker 
* bias problem 
* delayed feedback problem (user accepts after days)
* personalized random walk (for baseline)


# Search System 

### 1. Problem Formulation
* Clarifying questions
    - Is it a generalized search engine (like google) or specialized (like amazon product)?
    - What is the primary (business) objective of the search system?
    - What are the specific use cases and scenarios where it will be applied?
    - What are the system requirements (such as response time, accuracy, scalability, and integration with existing systems or platforms)?
    - What is the expected scale of the system in terms of data and user interactions?
    - Is their any data available? What format? 
    - Personalized? not required 
    - How many languages needs to be supported?
    - What types of items (products) are available on the platform, and what attributes are associated with them?
    - What are the common user search behaviors and patterns? Do users frequently use filters, sort options, or advanced search features?
    - Are there specific search-related challenges unique to the use case (e-commerce)? such as handling product availability, pricing, and customer reviews?

    
* Use case(s) and business goal
  * Use case: user enters text query into search box, system shows the most relevant items (products) 
  * business goal: increase CTR, conversion rate, etc  
* Requirements
  * response time, accuracy, scalability (50M DAU)
* Constraints
  * budget limitations, hardware limitations, or legal and privacy constraints
* Data: sources and availability
  * Sources:  
  * 
* Assumptions
* ML formulation: 
  * ML Objective: retrieve items that are most relevant to a text query  
    * we can define relevance as weighted summary of click, successful session, conversion, etc. 
  * ML I/O: I: text query from a user, O: ranked list of most relevant items on an e-commerce platform  
  * ML category: MM input search system -> retrieval and ranking 
    * ranking: MM input -> multi-label classification (click, success, convert, etc)
    * we can use a multi-task classifier 
   
### 2. Metrics  
- Offline
  - Precision@k, Recall@k, MRR, mAP, NDCG  
  - we choose NDCG (non-binary relevance)
- Online 
  - CTR: problem: doesn't track relevancy, click baits  
  - success session rate: dwell time > T or add to cart 
  - total dwell time 
  - conversion rate 

### 3. Architectural Components  
* Multimodal search (text, photo, video) for product content from text query: 
* Multi-layer architecture 
  * Query Understanding -> Candidate generation -> stage 1 Ranker -> stage 2 Ranker -> Blender -> Filter 
* Query understanding 
  * spell checker 
  * query normalization 
  * query expansion (e.g. add alternative) / relaxation (e.g. remove "good")
  * Intent/Domain classification 
* Candidate generation 
  * focus on recall, millions/billions into 10Ks 
* Ranking 
  * ML based 
  * multi-stage ranker: if more than 10k items to select from or QPS > 10k  
  * 100k items: stage 1 (liner model) -> stage 2 (DNN model) -> 500 items
* Blender: 
  * outputs a SERP (search engine result page)
  * blends results from multiple sources e.g. textual (inverted index, semantic) search, visual search, etc. 

#### Retrieval 
* from 100 B to 100k 
* IR: compares query text with document text 
* Document types: 
  * item (product) title 
  * item description 
  * item reviews 
  * item category 
* inverted index: 
  * index DS, mapping from words into their locations in a set of documents (e.g. ABC -> documents 1, 7)
* after query expansion (e.g. black pants into black and pants or suit-pants or trousers etc), do a search in inverted index db and find relevant items with relevance score 
* relevance score 
  * weighted linear combination of: 
    * terms match (e.g. TF-IDF score)(e.g. w = 0.5), 
    * item popularity (e.g. no of reviews, or bought) (e.g. w=0.125), 
    * intent match score (e.g. 0.125/2), 
    * domain match score,  
    * personalization score (e.g. age, gender, location, interests) 

#### Ranking: 
* see the next sections. 
<!-- 
- Visual search system 
  - Text query -> videos (based on similarity of text and visual content) 
  - Two tower embedding architecture (video and text_query encoders)
- Textual search system 
  - search for most similar titles, descs, and tags  w/ text query 
  - we can use Inverted Index (e.g. elastic search) for efficient full text search 
    - An inverted index is a data structure that maps terms (words) to the documents or locations where they appear, enabling efficient text-based document retrieval, commonly used in search engines. -->

### 4. Data Collection and Preparation
- Data sources: 
  - Users 
  - Queries 
  - Items (products)
  - Context 
- Labeling: 
  - use online user engagement data to generate positive and negative labels 
   
<!-- We use provided annotated data in the format of <video_id, query>.  -->
### 5. Feature Engineering
* Feature selection 
  * User: 
    * ID, username, 
    * Demographics (age, gender, location)
    * User interaction history (click rate, purchase rate, etc)
    * User interests (e.g. categories)
  * Context: 
    * device, 
    * time of the day, 
    * recent hype results 
    * previous queries 
  * Query features: 
    * query historical engagement (by other users)
    * query intent / domain 
    * query embeddings 
  * Item (product) features 
    * Title (exact text + embeddings)
    * Description (exact text + embeddings)
    * Reviews data (avg reviews, no of reviews, review textual data (text + embeddings)) 
    * category 
    * page rank 
    * engagement radius 
  * User-Item(product) features 
    * distance (e.g. for shipment)
    * historical engagement by the user (e.g. document type)
  * Query-Item(product) features
    * text match (title, description, category)
    * unigram or bigram search (title, description, category) - TF-IDF score 
    * historical engagement (e.g. click rate of Item for that query)
    * 
<!-- - Preprocessing unstructured data 
  - Text pre-processing : normalization, tokenization, token to ids
  - Video preprocessing: decode into frames -> sample -> resize -> scale, normalize, color correct  -->

### 6. Model Development and Offline Evaluation
#### Ranking 

* Model Selection  
  * Two options:
    * Pointwise LTR model: <user, item> -> relevance score 
      * approximate it as a binary classification problem p(relevant)
    * Pairwise LTR model: <user, item1, item2> -> item1 score > item2 score ?
      * loss function if the predicted order is correct 
      * more natural to ranking, more complicated 
  * Multi - Stage ranking 
    * 100k items (focus on recall) -> 500 items (focus on precision) -> 500 items in correct order   
    * Stage 1: We use a pointwise LTR -> binary classifier 
      * latency: microseconds 
      * suggestion: LR or small MART (multiple additive regression trees)
      * use ROC AUC for metric
    * Stage 2: Pairwise LTR model 
      * Two options (choose based on train data availability and capacity):
        * LambdaMART: a variation of MART, obj fcn changed to improve pairwise ranking  
        * LambdaRank: NN based model, pairwise loss (minimize inversions in ranking)
      * use NDCG for metric 

* Training Dataset
  * Pointwise approach 
    * positive samples: user engaged (e.g. click, spent time > T, add to cart, purchased)
    * negative samples: no engagement by the user + random negative samples e.g. from pages 10 and beyond
    * 5 million Q/day -> one positive one negative sample from each query -> 10 million samples a day 
    * use a whole week's data at least to capture daily patterns 
      * capturing and dealing with seasonal and holiday data 
    * train-valid/test split: 70/30 (of 70 million)
    * temporal affect: e.g. use 3 weeks data: first 2/3 of weeks: train, last week valid / test 
  * Pairwise approach: 
    * ranks items according to their relative order, which is closer to the nature of ranking 
    * predict doc scores in a way that miimizes No of inversions in the final ranked result 
    * Two options for train data generation for pointwise approach
      * human raters: each human rates 10 results per 100K queries * 10 humans = 10M examples
        * expensive, doesn't scale 
      * online engagement data 
        * assign scores to each engagement type e.g. 
          * impression with no click -> label/score 0 
          * click only -> score 1 
          * spent time after click > T : score 2 
          * add to cart : score 3 
          * purchase: score 4  
  
  <!-- - Text encoders:  -->
    <!-- - Text -> Vector (Embeddings)  
    - Two approaches: 
      - Statistical (BoW, TF-IDF)
      - ML encoders (word2vec, transformer based e.g. BERT)  
    - We chose transformer based (BERT). 

  - Video encoders: 
    - Video-level
      - more expensive, but captures temporal understanding
      - Example: ViViT (Video Vision Transformer)
    - Frame-level (from sample frames and aggregate)
      - less expensive (training and serving speed, compute power) 
      - Example: ViT 
 -->

<!-- * Model Training   
  - contrastive learning (similar to visual search system).  -->

### 7. Prediction Service
<!-- - Visual search from text query 
  - text -> preprocess -> encoder -> embedding 
  - videos are indexed by their encoded embeddings 
  - search: using approximate nearest neighbor search (ANN)
- Textual search
  - using Elasticsearch (full text / fuzzy search)
- Fusion  
  - re-rank based on weighted sum of rel scores 
  - re-rank using a model 
- Re-ranking 
  - business level logic and policies  -->
- Re-ranking 
  - business level logic and policies  -->
    - filtering inappropriate items 
    - diversity (exploration/exploitation)
    - etc 
  - Two ways: 
    - rule based filters and aggregators 
    - ML model 
      - Binary Classification (P(inappropriate))
      - Data sources: human raters, user feedback (report, review)
      - Features: same as product features in ranker
      - Models: LR, MART, or DNN (depending on data size, capacity, experiments)
      - More details on harmful content classification 

### 8. Online Testing and Deployment  
### 9. Scaling, Monitoring, and Updates
### 10. Other talking points 
* Positional bias 



# Design a video recommendation system 

## 1. Problem Formulation 
User-video interaction 

Some existing data examples:  
* videos data
* User historic data 
* Recommendations data 
* Reviews 
<!-- * video features include:  
Url, 
types
name,
desc_snippet,
recent_reviews,
all_reviews,
release_date,
developer,
publisher,
popular_tag, -->

### Clarifying questions 
- Use case? Homepage?
  - Does user sends a text query as well?
- Business objective? 
  - Increase user engagement (play, like, click, share), purchase?, create a better ultimate gaming experience 
- Similar to previously played, or personalized for the user? Personalized for the user
- User locations? Worldwide (multiple languages) 
- User’s age group: 
- Do users have any favorite lists, play later, etc?
- How many videos? 100 million
- How many users? 100 million DAU
- Latency requirements  - 200msec?
- Data access 
  - Do we log and have access to any data? Can we build a dataset using user interactions ?
  - Do we have textual description of items? 
- can users become friends on the platform and do we wanna take that into account?
- Free or Paid?  




### ML objective

- Recommend most engaging (define) videos
  * Max. No. of clicks (clickbait)
  * Max. No. completed videos/sessions/levels (bias to shorter)
  * Max. total hours played ()
  * Max. No. of relevant items (proxy by user implicit/explicit reactions) -> more control over signals, not the above shortcomings

* Define relevance: e.g. like is relevant, or playing half of it is, …
* ML Objective: build dataset and model to predict the relevance score b/w user and a video
* I/O: I: user_id, O: ranked list of videos + relevance score
* ML category: Recommendation System

## 2. Metrics (Offline and Online) 

* Offline: 
    * precision @k, mAP, and diversity 
* Online: 
    * CTR, # of completed, # of purchased, total play time, total purchase, user feedback 

## 3. Architectural Components (MVP Logic) 
The main approaches used for personalized recommendation systems: 
* Content-based filtering: suggest items similar to those user found relevant (e.g. liked)
    * No need for interaction data, recommends new items to users (no item cold start)
    * Capture unique interests of users
    * New user cold start 
    * Needs domain knowledge 
* CF: Using user-user (user based CF) or item-item similarities (item based CF)
    * Pros
        * No domain knowledge 
        * Capture new areas of interest 
        * Faster than content (no content info needed)
    * Cons: 
        * Cold start problem (both user and item)
        * No niche interest 
* Hybrid 
    * Parallel hybrid: combine(CF results, content based)
    * Sequential: [CF based] -> Content based

What do we choose? 
We choose a sequential hybrid model (standard e.g. for video recommendation)

We follow  the three stage recommender system (funnel architecture) in order to meet latency requirements and eb able to scale the system to billions of items. 

```mermaid
   Candidate generation --> Ranking --> Re-ranking
```

In the first stage, we use a light model to retrive thousands of items from millions
In the second (ranking) stage, we focus on high precision using a powerful model. This will not impact serving speed much because it's only run on smaller subset of items. 

Candidate generation in practice comes from aggregation of different candidate generation models. Here we can assume three candidate generation modules: 

1. Candidate generation 1 (Relevance based)
2. Candidate generation 2 (Popularity)
3. Candidate generation 3 (Trending)  

where we use CF for candidate generation 1

We use content based modeling for ranking.

## 4. Data preparation 

Data Sources: 

1. Users (user profile, historical interactions):
     * User profile
       * User_id, username, age, gender, location (city, country), lang, timezone


2. videos (structures, metadata, video content - what is it?)
   - video_id, title, date, rating, expected_length?,  #reviews, language, tags, description, price, developer, publisher, level, #levels

3. User-video interactions:  
Historical interactions: Play, purchase, like, and search history, etc  
   - User_id, video_id, timestamp, interaction_type(purchase, play, like, impression, search), interaction_val, location


1. Context: time of the day, day of the week, device, OS

Type

- Removing duplicates 
- filling missing values 
- normalizing data.

### Labeling: 
For features in the form of <user, video> pairs -> labeling strategy based on explicit or implicit feedback 
e.g. "positive" if user liked the item explicitly or interacted (e.g. watched/played) at least for X (e.g. half of it).   
negative samples: sample from background distribution -> correct via importance smapling 

## 5. Feature engineering 

There are several machine learning features that can be extracted from videos. Here are some examples:

- video metadata features
- video state: e.g. the positions of players, the status of objects and obstacles, the time remaining, and the score.
- video mechanics: The rules and interactions that govern the video. 
- User engagement: e.g. the length of play sessions, frequency of play, and player retention rates.
- Social interactions: b/w players: to identify patterns of behavior, such as the formation of alliances, the sharing of resources, and the types of communication used between players.
- Player preferences: which video features are most popular among players, which can help inform video design decisions.
- Player behaviors: player movement patterns, the types of actions taken by players, and the strategies used to achieve objectives.


We select some important features as follows:

* video metadata features: 
  * video ID, 
  Duration, 
  Language, 
  Title, 
  Description,
  Genre/Category, 
  Tags,  
  Publisher(popularity, reviews), 
  Release date, 
  Ratings, 
  Reviews, 
  (video content ?)
video titles, genres, platforms, release dates, user ratings, and user reviews.



* User profile: 
  * User ID, Age, Gender, Language, City, Country 

* User-item historical features: 
  * User-item interactions 
    * Played, liked, impressions
    * purchase history (avg. price)
  * User search history 

* Context


### Feature representation: 

* Categorical data (video_id, user_id, language, city): Use embedding layers, learned during
training 
* Categorical_data(gender, age): one_hot
* Continuous variables: normalize, or bucketize and one-hot (e.g. price) 
* Text:(title, desc, tags): title/description use embeddings, pre-trained BERT, fine tune on video language?, tags: CBOW
* 
* video content embeddings? 

## 6. Model Development and Offline Evaluation 

### 6.1 Candidate Generation 

For candidate generation 1 (Relevance Based), we choose CF. 

For CF there are two embedding based modeling options: 
1. Matrix Factorization 
   * Pros: Training speed (only two matrices to learn), Serving speed (static learned embeddings)
   * Cons: only relies on user-item interactions (No user profile info e.g. language is used); new-user cold start problem 
2. Two tower neural network:
   * Pros: Accepts user features (user profile + user search history) -> better quality recommendation; handles new users 
   * Cons: Expensive training, serving speed     

We chose two-tower network here. 

#### Two-tower network
  * two encoder towers (user tower + encoder tower)
  * user tower encodes user features into user embeddings  $u$ 
  * item tower encodes item features into item embeddings   $v_i$
  * similarity $u$, $v_i$ is considered as a relevance score (ranking as classification problem)


#### Loss function: 
Minimize cross entropy for each positive label and sampled negative examples 

### 6.2 Ranking 
For Ranking stage, we prioritize precision over efficiency. We choose content based filtering. Choose a model that relies in item features.  
ML Obj options: 
   - max P(watch| U, C)
   - max expected total watch time 
   - multi-objective (multi-task learning: add corresponding losses)
  
Model Options: 
- FF NN (e.g. similar tower network to a tower network) + logistic regression 
- Deep Cross Network (DCN)

Features  

* Video ID embeddings (watched video embedding avg, impression video embedding), 
* Video historic
  *  No. of previous impressions, reviews, likes, etc
  *  Time features (e.g. time since last play), 
* Language embedding (user, item), 
* User profile 
* User Historic (e.g. search history)



### 6.3 Re-Ranking 
Re-ranks items by additional business criteria (filter, promote)  
We can use ML models for clickbait, harmful content, etc or use heuristics   
Examples: 
* Age restriction filter 
* Region restriction filter 
* Video freshness (promote fresh content)
* Deduplication 
* Fairness, bias, etc 




## 7. Prediction Service 
two-tower network inference: find the k-top most relevant items given a user ->  
It's a classic nearest neighbor problem -> use approximate nearest neighbor (ANN) algorithms  

## 8. Online Testing and Deployment  
Standard approaches as before.  
## 9. Scaling
The three stage candidate generation - ranking - re-ranking can be scaled well as described earlier. It also meets the requirements of speed (funnel architecture), precision(ranking component), and diversity (multiple candid generation). 

### Cold start problem: 
* new users: two tower architectures accepts new users and we can still use user profile info even with no interaction 
* new items: recommend to random users and collect some data - then fine tune the model using new data

### Training: 
We need to be able to fine tune the model 
### Exploration exploitation trade-off 
- Multi-armed bandit (an agent repeatedly selects an option and receives a reward/cost. The goal of to maximize its cumulative reward over time, while simultaneously learning which options are most valuable.)
### Other Extensions: 
* [Multi-task learning](https://daiwk.github.io/assets/youtube-multitask.pdf)
  * Includes a shared feature extractor that is trained jointly with multiple prediction heads, each of which is responsible for predicting a different aspect of user behavior, such as click-through rate, watch time, and view count. The model is trained using a combination of supervised and unsupervised learning techniques, including cross-entropy loss, pairwise ranking loss, and self-supervised contrastive learning.
* Positional bias (detection and correction) 
* Selection bias (detection and correction)
* Add negative feedback (dislike)
* Locality preservation: 
  * Use sequential user behavior info (CBOW model)
* effect of seasonality 
* what if we only have a query and personal (item, provider) history? 
  * item embeddings, provider embeddings, query embeddings 
  * we can build a query-aware attention mechanism that computes 

### More resources 

* [Content-based](https://www.kaggle.com/code/fetenbasak/content-based-recommendation-video-recommender), [NLP analysis](https://www.kaggle.com/code/greentearus/steam-reviews-nlp-analysis), [Collaborative Denoising AE](https://www.kaggle.com/code/krsnewwave/collaborative-denoising-autoencoder-steam)  
* [User-based CF, item-based CF and MF](https://github.com/manandesai/video-recommendation-engine) ([github](https://github.com/manandesai/video-recommendation-engine/blob/main/recommenders.ipynb))
* [CF and content based](https://github.com/AudreyGermain/video-Recommendation-System) 

