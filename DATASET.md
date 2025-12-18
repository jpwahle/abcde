# ABCDE Dataset Documentation

**ABCDE** (Age-Based Corpus of Demographic Expressions) contains linguistic and demographic information extracted from multiple sources including Reddit posts, Twitter/X posts (via [TUSC](https://aclanthology.org/2022.lrec-1.442.pdf)), AI-generated text, blog posts, and Google Books Ngrams.

## Dataset Statistics

### Users with Self-Identified Age

| Dataset | Time Period | Number of Users |
|---------|-------------|-----------------|
| Reddit | 2010-2022 | 1,472,787 |
| TUSC-city | 2020-2021 | 13,220 |
| TUSC-country | 2015-2021 | 536 |

### Posts from Users with Self-Identified Age

| Dataset | Time Period | Number of Posts |
|---------|-------------|-----------------|
| Reddit | 2010-2022 | 37,989,673 |
| TUSC-city | 2020-2021 | 1,987,993 |
| TUSC-country | 2015-2021 | 12,324 |

## Dataset Files

### Reddit Dataset (`reddit/`)
- **reddit_users.tsv**: Contains Reddit users who self-identified their age with demographic extractions
- **reddit_users_posts.tsv**: Contains all posts from self-identified users with linguistic features

### TUSC (Twitter/X) Datasets (`tusc/`)
- **city_users.tsv**: Contains Twitter/X users who self-identified their age (city-level location)
- **city_user_posts.tsv**: Contains all posts from self-identified users with linguistic features (city-level)
- **country_users.tsv**: Contains Twitter/X users who self-identified their age (country-level location)
- **country_user_posts.tsv**: Contains all posts from self-identified users with linguistic features (country-level)

### AI-Generated Text Dataset (`ai-gen/`)
Contains AI-generated text from various sources with linguistic features:
- **anthropic_persuasiveness_data_features.tsv**: Persuasive text samples from Anthropic
- **apt-paraphrase-dataset-gpt-3_features.tsv**: GPT-3 paraphrases
- **general_thoughts_430k_data_features.tsv**: General AI thoughts/reflections
- **hh-rlhf_data_features.tsv**: Helpful/Harmless RLHF data
- **lmsys_data_features.tsv**: LMSYS chatbot arena conversations
- **luar_lwd_data_features.tsv**: LUAR linguistic writeprint data
- **m4_data_features.tsv**: M4 dataset samples
- **mage_data_features.tsv**: MAGE dataset samples
- **pippa_data_features.tsv**: PIPPA conversational AI data
- **prism_data_features.tsv**: PRISM dataset samples
- **raid_data_features.tsv**: RAID AI detection dataset
- **reasoning_shield_data_features.tsv**: Reasoning shield data
- **star1_data_features.tsv**: STAR1 dataset samples
- **tinystories_data_features.tsv**: TinyStories generated content
- **wildchat_data_features.tsv**: WildChat conversational data

### Blog Posts Dataset (`blogs/`)
Blog posts organized by tier groups, each containing:
- **spinner_blog_posts_features.tsv**: Blog posts with linguistic features
  - Tier groups: 2-13 (representing different author cohorts)

### Google Books Ngrams Dataset (`books/`)
- **googlebooks-eng-fiction-top1M-5gram.tsv**: Top 1 million 5-grams from English fiction with linguistic features

## Dataset Construction Process

### 1. Data Sources
- **Reddit**: JSON Lines files containing Reddit posts from 2010-2022 from [Pushshift](https://archive.org/download/pushshift_reddit_200506_to_202212/reddit/submissions)
- **TUSC**: Parquet files containing geolocated Twitter/X posts from [TUSC](https://github.com/tusc-project/tusc-dataset)
- **Google Books Ngrams (Fiction)**: 5-grams from the [Google Books Ngrams dataset](https://storage.googleapis.com/books/ngrams/books/datasetsv2.html) (v20120701) with format "ngram TAB year TAB match_count TAB book_count NEWLINE"
- **AI-Generated Text**: Various datasets including RAID, WildChat, LMSYS, PIPPA, and others

### 2. Processing Pipeline

The dataset was constructed using a two-stage pipeline:

#### Stage 1: Self-Identification Detection
- Scans posts/tweets to find users who self-identify their age using regex patterns to detect age mentions
- Resolves multiple age mentions to determine birth year
- Outputs user files with demographic information

#### Stage 2: Feature Extraction
- Collects all posts from self-identified users
- Applies feature extraction using various lexicons
- Computes age at post time based on birth year
- Outputs post files with all features

### 3. Filtering Criteria
- **Text length**: 5-1000 words
- **Age range**: 13-100 years old
- **Excluded authors**: [deleted], AutoModerator, Bot (Reddit only)
- **Valid self-identification**: Must match one of the regex patterns
- **Remove posts marked as adult material** (over_18 flag, Reddit only)
- **Remove posts with title but no body text** (Reddit only)
- **Remove promoted/advertised posts** (Reddit only)

## Age Extraction

### Regex Patterns Used

The system uses 6 regex patterns to detect age self-identification:

1. **Direct age statement**: `\bI(?:\s+am|'m)\s+(\d{1,2})\s+years?\s+old\b`
   - Example: "I am 25 years old", "I'm 30 year old"

2. **Age with contextual boundaries**: `\bI(?:\s+am|'m)\s+(\d{1,2})(?=\s*(?:$|[,.!?;:\-]|(?:and|but|so|yet)\s))`
   - Example: "I am 25.", "I'm 30, and...", "I am 25 but..."

3. **Birth year (4-digit)**: `\bI(?:\s+was|\s+am|'m)\s+born\s+in\s+(19\d{2}|20(?:0\d|1\d|2[0-4]))\b`
   - Example: "I was born in 1998", "I am born in 2005"

4. **Birth year (2-digit with apostrophe)**: `\bI(?:\s+was|\s+am|'m)\s+born\s+in\s+'(\d{2})\b`
   - Example: "I was born in '98", "I'm born in '05"

5. **Birth date (full format)**: `\bI\s+was\s+born\s+on\s+(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?(19\d{2}|20(?:0\d|1\d|2[0-4]))\b`
   - Example: "I was born on 15 March 1998", "I was born on March 15th, 1998"

6. **Birth date (numeric format)**: `\bI\s+was\s+born\s+on\s+\d{1,2}[/\-]\d{1,2}[/\-](19\d{2}|20(?:0\d|1\d|2[0-4]))\b`
   - Example: "I was born on 03/15/1998", "I was born on 15-03-1998"

### False Positive Prevention
- Word boundaries ensure complete word matches
- Contextual boundaries for pattern 2 (punctuation or conjunctions)
- Year ranges limited to 1900-2024
- Age filtering: only 13-100 years old accepted
- First-person requirement ("I") ensures self-identification

### Age Resolution Algorithm
1. Extract all age/birthyear mentions from text
2. Convert ages to birth years (post year - age)
3. Filter out ages below 13 during conversion
4. Cluster similar birth years (within 2 years)
5. Weight birth years (1.0) higher than ages (0.8)
6. Select cluster with highest score (weight sum + count × 0.1)
7. Compute weighted average as final birth year
8. Calculate resolved age and filter if not between 13-100

## Lexicons Used

### NRC Lexicons
- **NRC VAD Lexicon** (Version 1, July 2018)
  - Contains valence, arousal, and dominance scores (0-1) for words
  - Source: [NRC Word-Emotion Association Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html)

- **NRC Emotion Lexicon** (Version 0.92, July 2011)
  - Maps words to 8 emotions (anger, anticipation, disgust, fear, joy, sadness, surprise, trust) and 2 sentiments (positive, negative)
  - Source: [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)

- **NRC WorryWords Lexicon** (Anxiety/Calmness)
  - Contains anxiety scores from -3 (very calm) to +3 (very anxious)
  - Source: [NRC Word-Worry Association Lexicon](http://saifmohammad.com/worrywords.html)

- **NRC MoralTrust Lexicon** (Version: Jan 5, 2025)
  - Contains moral trustworthiness scores
  - Source: [NRC Lexicons](https://saifmohammad.com/WebPages/lexicons.html)

- **NRC SocialWarmth Lexicon** (Version: Jan 5, 2025)
  - Contains social warmth scores
  - Source: [NRC Lexicons](https://saifmohammad.com/WebPages/lexicons.html)

- **NRC CombinedWarmth Lexicon** (Version: Jan 5, 2025)
  - Contains combined warmth scores
  - Source: [NRC Lexicons](https://saifmohammad.com/WebPages/lexicons.html)

### Other Lexicons
- **ENG Tenses Lexicon** (Version 3, April 2022)
  - Maps words to their grammatical forms (past, present, etc.)
  - Source: [UniMorph English](https://github.com/unimorph/eng)

- **Body Part Words**: Union of two sources:
  - [Collins Dictionary Body Parts List](https://www.collinsdictionary.com/us/word-lists/body-parts-of-the-body)
  - [Enchanted Learning Body Parts List](https://www.enchantedlearning.com/wordlist/body.shtml)

- **Cognitive/Thinking Words Lexicon**
  - Categorized list of cognitive and thinking-related verbs
  - 12 categories covering different types of cognitive processes

## Feature Descriptions

### Demographic Features (DMG prefix)
- **Author**: User ID/username
- **DMGMajorityBirthyear**: Resolved birth year from self-identification
- **DMGRawBirthyearExtractions**: Raw extracted age/year values
- **DMGRawExtractedAge**: Raw age mentions extracted from text
- **DMGRawExtractedGender**: Gender self-identifications extracted from text
- **DMGRawExtractedCity**: City names extracted from text
- **DMGCountryMappedFromExtractedCity**: Country derived from extracted city using GeoNames database
- **DMGRawExtractedCountry**: Country names extracted directly from text
- **DMGRawExtractedReligion**: Religion mentions extracted from text
- **DMGMainReligionMappedFromExtractedReligion**: Primary religion mapped from extracted text
- **DMGMainCategoryMappedFromExtractedReligion**: Religion category (e.g., Christianity, Islam, etc.)
- **DMGRawExtractedOccupation**: Occupation mentions extracted from text
- **DMGSOCTitleMappedFromExtractedOccupation**: Standard Occupational Classification (SOC) title mapped from extracted occupation
- **DMGAgeAtPost**: Age when the post was created

### Post Metadata

#### Reddit-specific
- **PostID**: Unique post identifier
- **PostCreatedUtc**: Unix timestamp of post creation
- **PostSubreddit**: Subreddit name
- **PostTitle**: Post title
- **PostSelftext**: Post body content
- **PostScore**: Reddit score (upvotes minus downvotes)
- **PostNumComments**: Number of comments on the post
- **PostPermalink**: Permanent link to the post
- **PostUrl**: URL (if link post)
- **PostMediaPath**: Path to associated media (if any)

#### TUSC-specific
- **PostText**: Tweet content
- **PostCreatedAt**: Timestamp of tweet creation
- **PostYear**: Year of post
- **PostMonth**: Month of post
- **PostCity**: City-level location (city dataset)
- **PostCountry**: Country-level location (country dataset)
- **PostMyCountry**: User's country
- **PostPlace**: Twitter place name
- **PostPlaceID**: Twitter place ID
- **PostPlaceType**: Type of place (city, admin, etc.)

#### AI-Generated Text Metadata
- **source**: Dataset source name
- **type**: Content type classification
- **conv_id**: Conversation ID
- **user_prompt**: User input that generated the response
- **turn_in_conv**: Turn number in conversation
- **timestamp**: When the content was generated
- **model**: AI model that generated the text
- **ai_text**: The generated text content
- **is_winner**: (LMSYS) Whether this was the winning response
- **domain**: Content domain/category
- **title**: Title of the content
- **bot_id**: (PIPPA) Bot identifier
- **state**: (WildChat) User state location
- **country**: (WildChat) User country location
- **rid**: Record ID
- **decoding**: Decoding strategy used
- **repetition_penalty**: Repetition penalty applied

#### Blog Post Metadata
- **file_path**: Path to source file
- **title**: Blog post title
- **link**: URL link
- **guid**: Globally unique identifier
- **pubDate**: Publication date
- **description_raw**: Raw description text
- **description**: Processed description
- **categories**: Blog categories/tags

#### Google Books Ngram Metadata
- **ngram**: The 5-gram text
- **year**: Year of occurrence
- **match_count**: Number of times ngram appeared
- **book_count**: Number of books containing the ngram

### Body Part Mentions (BPM prefix)
- **HasBPM**: Boolean - any body part found in text
- **MyBPM**: Body parts mentioned after "my" (e.g., "my head")
- **YourBPM**: Body parts mentioned after "your"
- **HerBPM**: Body parts mentioned after "her"
- **HisBPM**: Body parts mentioned after "his"
- **TheirBPM**: Body parts mentioned after "their"

### Pronoun Features (PRN prefix)
Binary flags for presence of pronouns:

**First Person Singular:**
- **PRNHasI**: Contains "I"
- **PRNHasMe**: Contains "me"
- **PRNHasMy**: Contains "my"
- **PRNHasMine**: Contains "mine"

**First Person Plural:**
- **PRNHasWe**: Contains "we"
- **PRNHasOur**: Contains "our"
- **PRNHasOurs**: Contains "ours"

**Second Person:**
- **PRNHasYou**: Contains "you"
- **PRNHasYour**: Contains "your"
- **PRNHasYours**: Contains "yours"

**Third Person Feminine:**
- **PRNHasShe**: Contains "she"
- **PRNHasHer**: Contains "her"
- **PRNHasHers**: Contains "hers"

**Third Person Masculine:**
- **PRNHasHe**: Contains "he"
- **PRNHasHim**: Contains "him"
- **PRNHasHis**: Contains "his"

**Third Person Plural/Neutral:**
- **PRNHasThey**: Contains "they"
- **PRNHasThem**: Contains "them"
- **PRNHasTheir**: Contains "their"
- **PRNHasTheirs**: Contains "theirs"

### Temporal/Tense Features (TIME prefix)
Features based on verb tense analysis using the UniMorph English lexicon:

- **TIMEHasPastVerb**: Boolean - text contains at least one past tense verb
- **TIMECountPastVerbs**: Count of past tense verbs in text
- **TIMEHasPresentVerb**: Boolean - text contains at least one present tense verb
- **TIMECountPresentVerbs**: Count of present tense verbs in text
- **TIMEHasFutureModal**: Boolean - text contains future modal verbs (will, shall, etc.)
- **TIMECountFutureModals**: Count of future modal verbs
- **TIMEHasPresentNoFuture**: Boolean - has present tense but no future reference
- **TIMEHasFutureReference**: Boolean - text contains future-oriented language

### Cognitive/Thinking Word Features (COG prefix)
Binary flags indicating presence of words from 12 cognitive categories:

- **COGHasAnalyzingEvaluatingWord**: Analyzing & evaluating words (analyze, assess, evaluate, investigate, critique, etc.)
- **COGHasCreativityIdeationWord**: Creativity & ideation words (brainstorm, imagine, create, innovate, visualize, etc.)
- **COGHasGeneralCognitionWord**: General cognition words (contemplate, deliberate, focus, reflect, reason, etc.)
- **COGHasLearningUnderstandingWord**: Learning & understanding words (learn, understand, comprehend, grasp, study, etc.)
- **COGHasDecisionMakingJudgingWord**: Decision making & judging words (decide, choose, judge, determine, calculate, etc.)
- **COGHasProblemSolvingWord**: Problem solving words (solve, plan, strategize, troubleshoot, revise)
- **COGHasHigher-OrderThinkingWord**: Higher-order thinking words (abstract, categorize, synthesize, hypothesize, interpret, etc.)
- **COGHasConfusedorUncertainThinkingWord**: Confused/uncertain thinking words (doubt, self-question)
- **COGHasMemoryRecallWord**: Memory & recall words (remember, recall, forget, memorize, retrieve, etc.)
- **COGHasPerceptionObservationWord**: Perception & observation words (notice, observe, recognize, identify, detect, etc.)
- **COGHasPredictionForecastingWord**: Prediction & forecasting words (predict, anticipate, forecast, project, forethink)
- **COGHasExplanationArticulationWord**: Explanation & articulation words (explain, describe, define, elaborate, discuss, etc.)

### NRC VAD Features
Valence-Arousal-Dominance scores from the NRC VAD Lexicon:

**Valence** (emotional positivity/negativity, 0-1 scale):
- **NRCAvgValence**: Average valence score across all matched words
- **NRCHasHighValenceWord**: Boolean - contains words with high valence (≥0.8)
- **NRCHasLowValenceWord**: Boolean - contains words with low valence (≤0.2)
- **NRCCountHighValenceWords**: Count of high valence words
- **NRCCountLowValenceWords**: Count of low valence words

**Arousal** (emotional intensity/activation, 0-1 scale):
- **NRCAvgArousal**: Average arousal score
- **NRCHasHighArousalWord**: Boolean - contains high arousal words (≥0.8)
- **NRCHasLowArousalWord**: Boolean - contains low arousal words (≤0.2)
- **NRCCountHighArousalWords**: Count of high arousal words
- **NRCCountLowArousalWords**: Count of low arousal words

**Dominance** (sense of control, 0-1 scale):
- **NRCAvgDominance**: Average dominance score
- **NRCHasHighDominanceWord**: Boolean - contains high dominance words (≥0.8)
- **NRCHasLowDominanceWord**: Boolean - contains low dominance words (≤0.2)
- **NRCCountHighDominanceWords**: Count of high dominance words
- **NRCCountLowDominanceWords**: Count of low dominance words

### NRC Emotion Features
Discrete emotion detection from the NRC Emotion Lexicon:

**Eight Basic Emotions:**
- **NRCHasAngerWord** / **NRCCountAngerWords**: Anger-associated words
- **NRCHasAnticipationWord** / **NRCCountAnticipationWords**: Anticipation-associated words
- **NRCHasDisgustWord** / **NRCCountDisgustWords**: Disgust-associated words
- **NRCHasFearWord** / **NRCCountFearWords**: Fear-associated words
- **NRCHasJoyWord** / **NRCCountJoyWords**: Joy-associated words
- **NRCHasSadnessWord** / **NRCCountSadnessWords**: Sadness-associated words
- **NRCHasSurpriseWord** / **NRCCountSurpriseWords**: Surprise-associated words
- **NRCHasTrustWord** / **NRCCountTrustWords**: Trust-associated words

**Sentiment:**
- **NRCHasPositiveWord** / **NRCCountPositiveWords**: Positive sentiment words
- **NRCHasNegativeWord** / **NRCCountNegativeWords**: Negative sentiment words

### NRC WorryWords Features
Anxiety and calmness detection from the NRC WorryWords Lexicon:

- **NRCHasAnxietyWord**: Boolean - contains anxiety-associated words
- **NRCHasCalmnessWord**: Boolean - contains calmness-associated words
- **NRCAvgAnxiety**: Average anxiety score (positive = anxious)
- **NRCAvgCalmness**: Average calmness score (positive = calm)
- **NRCHasHighAnxietyWord**: Boolean - contains highly anxious words (score ≥2)
- **NRCCountHighAnxietyWords**: Count of highly anxious words
- **NRCHasHighCalmnessWord**: Boolean - contains highly calm words (score ≤-2)
- **NRCCountHighCalmnessWords**: Count of highly calm words

### NRC Moral/Social/Warmth Features

**Moral Trust Features** (perceived trustworthiness):
- **NRCHasHighMoralTrustWord**: Boolean - high moral trust words (OrdinalClass=3)
- **NRCCountHighMoralTrustWord**: Count of high moral trust words
- **NRCHasLowMoralTrustWord**: Boolean - low moral trust words (OrdinalClass=-3)
- **NRCCountLowMoralTrustWord**: Count of low moral trust words
- **NRCAvgMoralTrustWord**: Average moral trust score

**Social Warmth Features** (interpersonal warmth):
- **NRCHasHighSocialWarmthWord**: Boolean - high social warmth words (OrdinalClass=3)
- **NRCCountHighSocialWarmthWord**: Count of high social warmth words
- **NRCHasLowSocialWarmthWord**: Boolean - low social warmth words (OrdinalClass=-3)
- **NRCCountLowSocialWarmthWord**: Count of low social warmth words
- **NRCAvgSocialWarmthWord**: Average social warmth score

**Combined Warmth Features** (overall warmth):
- **NRCHasHighWarmthWord**: Boolean - high warmth words (OrdinalClass=3)
- **NRCCountHighWarmthWord**: Count of high warmth words
- **NRCHasLowWarmthWord**: Boolean - low warmth words (OrdinalClass=-3)
- **NRCCountLowWarmthWord**: Count of low warmth words
- **NRCAvgWarmthWord**: Average warmth score

### Basic Text Statistics
- **WordCount**: Total word count in the text
