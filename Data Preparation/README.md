# Data Preparation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tangnatta/LaughOlogy-Links-Brain-Hackathon-2023/blob/main/Data%20Preparation/Brain_hackathon_Data_Prep.ipynb)

## Data Sources

Joke dataset from

- [A dataset of English plaintext jokes.](https://github.com/taivop/joke-dataset) (Pungas, Taivo 2017)
- [The rJokes Dataset: a Large Scale Humor Collection](https://aclanthology.org/2020.lrec-1.753) (Weller & Seppi, LREC 2020)

## Data Cleaning

- Filter out jokes with less than 2 words and null jokes.
- Joke Deduplication
  - Remove exact duplicates.
  - Remove duplicates with similarity by generating a vector from jokes with the Universal Sentence Encoder, then find the cosine similarity (cosine similarity approximated with the [annoy](https://github.com/spotify/annoy) library).

## Funny and Not Funny Jokes Separation

- Using score thresholding separation (score > 24 is funny and score <= 24 is not funny), which is labeled in the `funny` column in the dataset

## Generating similar non-humor text from funny jokes

- Using Gemini-Pro model via the Google AI API
