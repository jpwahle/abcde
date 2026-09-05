# Citation guide

Cite [ABCDE](README.md#citation) whenever using the dataset or code. Also cite the work behind every feature family and source dataset used. Requirements are cumulative, including when selecting or filtering records by a feature. Source papers and dataset cards linked below provide citation metadata and terms of use.

## Feature families

| Features used | Additional citations |
|---------------|----------------------|
| Age, birth year, age-selected data (`DMGAgeAtPost`, `DMGMajorityBirthyear`, age extractions) | Daniela Teodorescu, Jan Philip Wahle, and Saif M. Mohammad (2026). [Age and Affect in Language: How Emotion Expression on Social Media Varies Across Adulthood](https://aclanthology.org/2026.cas-1.15/). CAS @ LREC 2026, pp. 175–189. |
| VAD (`NRCAvgValence`, `NRCAvgArousal`, `NRCAvgDominance`, and corresponding high/low flags and counts) | Saif M. Mohammad (2018). [Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words](https://aclanthology.org/P18-1017/). **And** Saif M. Mohammad (2025). [NRC VAD Lexicon v2: Norms for Valence, Arousal, and Dominance for over 55k English Terms](https://arxiv.org/abs/2503.23547). Cite both. |
| NRC emotion and sentiment flags/counts (anger, anticipation, disgust, fear, joy, sadness, surprise, trust, positive, negative) | Saif Mohammad and Peter Turney (2013), *Crowdsourcing a Word-Emotion Association Lexicon*, and Mohammad and Turney (2010), *Emotions Evoked by Common Words and Phrases*. Both are listed on the [NRC Emotion Lexicon citation page](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm). |
| Anxiety/calmness (`NRC*Anxiety*`, `NRC*Calmness*`) | Saif M. Mohammad (2024). [WorryWords: Norms of Anxiety Association for over 44k English Words](https://aclanthology.org/2024.emnlp-main.910/). |
| Moral trustworthiness, social warmth, combined warmth (`NRC*MoralTrust*`, `NRC*SocialWarmth*`, `NRC*Warmth*`) | Saif M. Mohammad (2025). [Words of Warmth: Trust and Sociability Norms for over 26k English Words](https://aclanthology.org/2025.acl-long.922/). If using dominance as competence, also cite both VAD papers. |
| Body-part mentions (`HasBPM`, `MyBPM`, `YourBPM`, `HerBPM`, `HisBPM`, `TheirBPM`) | Sophie Wu, Jan Philip Wahle, and Saif M. Mohammad (2025). [The Language of Interoception: Examining Embodiment and Emotion Through a Corpus of Body Part Mentions](https://aclanthology.org/2025.findings-emnlp.1269/). Word-list sources are documented in [DATASET.md](DATASET.md#other-lexicons). |
| Verb-tense features (`TIME*`) | Cite the [UniMorph English resource](https://github.com/unimorph/eng) and the publication specified by that resource for the version used. ABCDE's word-list version is recorded in [DATASET.md](DATASET.md#other-lexicons). |

The citations above also apply to features with a text-field prefix, such as `model_reasoning_NRCAvgValence`. Citation of the 2025 VAD paper does not imply that existing values were recomputed with VAD v2; the documented original lexicon version still applies.

## Source datasets

| Data used | Additional citation |
|-----------|---------------------|
| TUSC (`tusc/`, city or country) | Krishnapriya Vishnubhotla and Saif M. Mohammad (2022). [Tweet Emotion Dynamics: Emotion Word Usage in Tweets from US and Canada](https://aclanthology.org/2022.lrec-1.442/). |
| Reddit (`reddit/`) | Jason Baumgartner et al. (2020). [The Pushshift Reddit Dataset](https://ojs.aaai.org/index.php/ICWSM/article/view/7347). |
| Spinn3r blogs (`blogs/`) | Kevin Burton, Akshay Java, and Ian Soboroff (2009). *The ICWSM 2009 Spinn3r Dataset*. See the [official dataset citation](https://www.icwsm.org/2009/data/). |
| Google Books fiction 5-grams (`books/`) | Yuri Lin et al. (2012). [Syntactic Annotations for the Google Books NGram Corpus](https://aclanthology.org/P12-3029/). Identify the English Fiction subset and version 20120701. |

## AI-generated text

Cite the original compilation paper for **each** source you use, in addition to ABCDE and any applicable feature papers. Citing only the model that generated a response does not credit the dataset compilation. Keep the source/model metadata and follow each dataset's upstream attribution requirements. File names below are those in the HF snapshot; an internal Parquet release retains their stems.

| ABCDE file under `ai-gen/` | Compilation paper or resource |
|---------------------------|-------------------------------|
| `anthropic_persuasiveness_data_features.tsv` | Durmus et al. (2024). [Measuring the Persuasiveness of Language Models](https://www.anthropic.com/research/measuring-model-persuasiveness). |
| `apt-paraphrase-dataset-gpt-3_features.tsv` | Wahle et al. (2022). [How Large Language Models are Transforming Machine-Paraphrase Plagiarism](https://aclanthology.org/2022.emnlp-main.62/). |
| `general_thoughts_430k_data_features.tsv` | General Reasoning (2025). [GeneralThought-430K, now archived as GeneralThoughtArchive](https://huggingface.co/datasets/RJT1990/GeneralThoughtArchive). Cite the resource and its applicable upstream sources; the dataset card supplies provenance rather than a separate compilation paper. |
| `hh-rlhf_data_features.tsv` | Bai et al. (2022). [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862). |
| `lmsys_data_features.tsv` | Zheng et al. (2024). [LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset](https://arxiv.org/abs/2309.11998). |
| `luar_lwd_data_features.tsv` | Rivera Soto et al. (2024). [Few-Shot Detection of Machine-Generated Text using Style Representations](https://arxiv.org/abs/2401.06712). |
| `m4_data_features.tsv` | Wang et al. (2024). [M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection](https://aclanthology.org/2024.eacl-long.83/). |
| `mage_data_features.tsv` | Li et al. (2024). [MAGE: Machine-generated Text Detection in the Wild](https://aclanthology.org/2024.acl-long.3/). |
| `pippa_data_features.tsv` | Gosling, Dale, and Zheng (2023). [PIPPA: A Partially Synthetic Conversational Dataset](https://arxiv.org/abs/2308.05884). |
| `prism_data_features.tsv` | Kirk et al. (2024). [The PRISM Alignment Dataset](https://arxiv.org/abs/2404.16019). |
| `raid_data_features.tsv` | Dugan et al. (2024). [RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors](https://aclanthology.org/2024.acl-long.674/). |
| `reasoning_shield_data_features.tsv` | Li et al. (2025). [ReasoningShield: Content Safety Detection over Reasoning Traces of Large Reasoning Models](https://arxiv.org/abs/2505.17244). |
| `star1_data_features.tsv` | Wang et al. (2025). [STAR-1: Safer Alignment of Reasoning LLMs with 1K Data](https://arxiv.org/abs/2504.01903). |
| `tinystories_data_features.tsv` | Eldan and Li (2023). [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759). |
| `wildchat_data_features.tsv` | Zhao et al. (2024). [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://arxiv.org/abs/2405.01470). |

## Terms and versions

ABCDE's MIT license covers its code. It does not replace source-dataset licenses, lexicon terms, or citation requirements. Record the ABCDE release/revision and the features and source subsets used in your methods. For locally supplied unpublished resources, obtain the creators' citation instructions and permission before any public distribution of the resources or a release incorporating them.
