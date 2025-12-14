
#  data_preparation.py From phase 1 -----> phase 3


import os
import re
import pandas as pd
import numpy as np
from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import stanza


RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Initialize Stanza Arabic pipeline
stanza.download("ar")
nlp = stanza.Pipeline(
    "ar",
    processors="tokenize,mwt,pos,lemma,depparse",
    use_gpu=False
)


# PHASE 1 

def load_and_label_dataset():
    print("🔹 Loading dataset from HuggingFace...")
    dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

    ai_cols = [
        "allam_generated_abstract",
        "jais_generated_abstract",
        "llama_generated_abstract",
        "openai_generated_abstract"
    ]

    dfs = []

    for split_name, split_data in dataset.items():

        df = split_data.to_pandas()

        # Human   0
        human_df = pd.DataFrame({"text": df["original_abstract"], "label": 0})
        dfs.append(human_df)

        # AI   1
        for col in ai_cols:
            if col in df.columns:
                ai_df = pd.DataFrame({"text": df[col], "label": 1})
                dfs.append(ai_df)

    final_raw = pd.concat(dfs, ignore_index=True).dropna().reset_index(drop=True)
    final_raw.to_csv(RAW_PATH + "raw_combined_abstracts.csv", index=False, encoding="utf-8-sig")

    print(" raw_combined_abstracts.csv saved.")
    return final_raw


# PHASE 2 — CLEANING 

def clean_arabic_text(text):
    if not isinstance(text, str):
        return ""

    # Remove tatweel + diacritics
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)

    # Normalize alef & yaa
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)

    # Remove punctuation / non-Arabic
    text = re.sub(r"[^\u0600-\u06FF0-9\s]", " ", text)

    # Clean spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(df):
    
    df["text_processed"] = df["text"].apply(clean_arabic_text)
    df.to_csv(PROCESSED_PATH + "merged_processed.csv", index=False, encoding="utf-8-sig")
    print(" merged_processed.csv saved.")
    return df


# PHASE 3 — FEATURE ENGINEERING

PUNCS = set("،؛؟.!:")

def feature_num_digits(text):
    C = max(len(text), 1)
    return sum(c.isdigit() for c in text) / C


def feature_unique_punctuation(text):
    C = max(len(text), 1)
    return len({c for c in text if c in PUNCS}) / C


def extract_stanza_features(text):
    doc = nlp(text)
    adjectives = 0
    genitives = 0

    for sent in doc.sentences:
        for w in sent.words:
            if w.upos == "ADJ":
                adjectives += 1
            if w.deprel == "nmod":
                genitives += 1

    return adjectives, genitives


def gini(values):
    if len(values) == 0:
        return 0
    arr = np.sort(np.array(values))
    n = len(arr)
    cum = np.cumsum(arr)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def compute_features(df):
    print("🔹 Extracting features ONCE for entire dataset (faster)...")

    digits_f = []
    punct_f = []
    adj_f = []
    gen_f = []
    gini_f = []

    for text, processed in zip(df["text"], df["text_processed"]):

        digits_f.append(feature_num_digits(text))
        punct_f.append(feature_unique_punctuation(text))

        adj, gen = extract_stanza_features(processed)
        adj_f.append(adj)
        gen_f.append(gen)

        counts = Counter(processed.split())
        gini_f.append(gini(list(counts.values())))

    df["F3_digits_ratio"] = digits_f
    df["F24_unique_punct_ratio"] = punct_f
    df["F45_adjectives"] = adj_f
    df["F66_genitives"] = gen_f
    df["F87_gini"] = gini_f

    df.to_csv(PROCESSED_PATH + "all_features.csv", index=False, encoding="utf-8-sig")
    print(" all_features.csv saved.")

    return df



# SPLITTING 

def split_dataset(df):
    print(" Splitting dataset 70/15/15 AFTER feature extraction ...")

    train, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
    val, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["label"])

    train.to_csv(PROCESSED_PATH + "train_features.csv", index=False, encoding="utf-8-sig")
    val.to_csv(PROCESSED_PATH + "val_features.csv", index=False, encoding="utf-8-sig")
    test.to_csv(PROCESSED_PATH + "test_features.csv", index=False, encoding="utf-8-sig")

    print(" train_features.csv, val_features.csv, test_features.csv saved.")
    return train, val, test



#  PIPELINE

def run_pipeline():
    print("\n Starting FAST Data Preparation Pipeline...\n")

    raw_df = load_and_label_dataset()
    processed_df = preprocess_text(raw_df)

    df_with_features = compute_features(processed_df)

    train, val, test = split_dataset(df_with_features)

    print("\n FINISH !\n")
    return train, val, test



if __name__ == "__main__":
    run_pipeline()
