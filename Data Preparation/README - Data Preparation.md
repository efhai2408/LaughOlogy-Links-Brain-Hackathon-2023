# Data Preparation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tangnatta/LaughOlogy-Links-Brain-Hackathon-2023/blob/Data%20Preparation\Brain_hackathon_Data_Prep.ipynb)

## Data Sources

Joke dataset from

- [A dataset of English plaintext jokes.](https://github.com/taivop/joke-dataset) (Pungas, Taivo 2017)
- [The rJokes Dataset: a Large Scale Humor Collection](https://aclanthology.org/2020.lrec-1.753) (Weller & Seppi, LREC 2020)

## Data Cleaning

- Filter out jokes with less than 2 words and nul joke
- Joke Deduplication
  - Remove exact duplicates
  - Remove duplicates with similarity by generating a vector from jokes with the Universal Sentence Encoder, then find the cosine similarity (cosine similarity approximated with the [annoy](https://github.com/spotify/annoy) library).

## Funny and Not Funny Jokes Separation

- Using score thresholding separation (score > 24 is funny and score <= 24 is not funny) which is label in `funny` column in the dataset

## Generating simiral non-humor text from funny jokes

- Using Gemini-pro model via Google AI API
