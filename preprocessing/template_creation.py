import argparse
import audformat
import json
import numpy as np
import os
import pandas as pd
import tqdm
import typing
import yaml

from sklearn.preprocessing import (
    QuantileTransformer
)

# NOTE: Please extract features by features.py first
def create_templates(row: pd.Series, templates: typing.List[str]):
    captions = []
    if "pitch" in templates:
        if row["F0semitoneFrom27.5Hz_sma3nz_amean.quantile"] >= 0.7:
            captions.append("has a high pitch")
        elif row["F0semitoneFrom27.5Hz_sma3nz_amean.quantile"] <= 0.3:
            captions.append("has a low pitch")
        else:
            captions.append("has an average pitch")
            captions.append("has a normal pitch")

        if row["F0semitoneFrom27.5Hz_sma3nz_stddevNorm.quantile"] >= 0.7:
            captions.append("has a high pitch variation")
            captions.append("has a high pitch variance")
            captions.append("has a very unstable pitch")
            captions.append("has a very unstable phonation")
        elif row["F0semitoneFrom27.5Hz_sma3nz_stddevNorm.quantile"] <= 0.3:
            captions.append("has a low pitch variation")
            captions.append("has a low pitch variance")
            captions.append("has a very stable pitch")
            captions.append("has a very stable phonation")
        else:
            captions.append("has a normal pitch variation")

    if "intensity" in templates:
        if (row["equivalentSoundLevel_dBp.quantile"] >= 0.7) or (row["loudness_sma3_amean.quantile"] >= 0.7):
            captions.append("is loud")
            captions.append("has a high equivalent sound level")
            captions.append("sound pressure is elevated")
            captions.append("sound level is elevated")
        elif (row["equivalentSoundLevel_dBp.quantile"] <= 0.3) or (row["loudness_sma3_amean.quantile"] <= 0.3):
            captions.append("has a low equivalent sound level")
            captions.append("is quiet")
            captions.append("is almost silent")
        else:
            captions.append("loudness is just about right")
            captions.append("has a normal equivalent sound level")
            captions.append("has an average equivalent sound level")

    if "duration" in templates:
        if row["Duration.quantile"] >= 0.7:
            captions.append("is long")
            captions.append("has a long duration")
            captions.append("is a long sentence")
            captions.append("lasts a long time")
            captions.append("has a big duration")
        elif row["Duration.quantile"] <= 0.3:
            captions.append("is short")
            captions.append("has a short duration")
            captions.append("is a short sentence")
            captions.append("lasts a little time")
            captions.append("has a small duration")
        else:
            captions.append("is of average length")
            captions.append("is of average duration")
            captions.append("is neither long or short")
            captions.append("duration is medium")
    
    if "jitter" in templates:
        if row["jitterLocal_sma3nz_amean.quantile"] >= 0.7:
            captions.append("has a high jitter")
            if row["F0semitoneFrom27.5Hz_sma3nz_stddevNorm.quantile"] <= 0.7:
                captions.append("has a high jitter but a low pitch variance")
                captions.append("has a high jitter but not a high pitch variance")
                captions.append("has a high jitter but the pitch is stable")
        elif row["jitterLocal_sma3nz_amean.quantile"] <= 0.3:
            captions.append("has a low jitter")
            if row["F0semitoneFrom27.5Hz_sma3nz_stddevNorm.quantile"] >= 0.3:
                captions.append("has a low jitter but a high pitch variance")
                captions.append("has a low jitter but not a low pitch variance")
                captions.append("has a low jitter but the pitch is unstable")
        else:
            captions.append("has a normal jitter")

    if "shimmer" in templates:
        if row["shimmerLocaldB_sma3nz_amean.quantile"] >= 0.7:
            captions.append("has a high shimmer")
            if row["F0semitoneFrom27.5Hz_sma3nz_stddevNorm.quantile"] <= 0.7:
                captions.append("has a high shimmer but a low pitch variance")
                captions.append("has a high shimmer but not a high pitch variance")
                captions.append("has a high shimmer but the pitch is stable")
        elif row["shimmerLocaldB_sma3nz_amean.quantile"] <= 0.3:
            captions.append("has a low shimmer")
            if row["F0semitoneFrom27.5Hz_sma3nz_stddevNorm.quantile"] >= 0.3:
                captions.append("has a low shimmer but a high pitch variance")
                captions.append("has a low shimmer but not a low pitch variance")
                captions.append("has a low shimmer but the pitch is unstable")
        else:
            captions.append("has a normal shimmer")

    if "emotion" in templates:
        if row["emotion"] == "no_agreement":
            captions.append("annotators disagree")
            captions.append("annotators do not agree")
            captions.append("sample is ambiguous")
            captions.append("there is no clear emotion")
        else:
            captions.append(f'sentence is {row["emotion"]}')
            captions.append(f'this is a {row["emotion"]} instance')
            captions.append(f'emotion is {row["emotion"]}')
            captions.append(f'speaker is {row["emotion"]}')

    if "arousal" in templates:
        if row["arousal.quantile"] >= 0.7:
            captions.append("has high arousal")
            captions.append("speaker is aroused")
            if row["emotion"] != "no_agreement":
                captions.append(f'speaker is very {row["emotion"]}')
        elif row["arousal.quantile"] <= 0.3:
            captions.append("has low arousal")
            captions.append("speaker is calm")
            if row["emotion"] != "no_agreement":
                captions.append(f'speaker is not very {row["emotion"]}')
        else:
            captions.append("arousal is at an average level")

    if "dominance" in templates:
        if row["dominance.quantile"] >= 0.7:
            captions.append("has high dominance")
            captions.append("speaker appears to be dominant")
        elif row["dominance.quantile"] <= 0.3:
            captions.append("has low dominance")
        else:
            captions.append("dominance is at an average level")

    if "valence" in templates:
        if row["valence.quantile"] >= 0.7:
            captions.append("has high valence")
            captions.append("speaker appears to be in a good mood")
        elif row["valence.quantile"] <= 0.3:
            captions.append("has low valence")
            captions.append("speaker appears to be in a bad mood")
        else:
            captions.append("valence is at an average level")

    if "gender" in templates:
        if row["gender"] == row["gender"]:
            captions.append(f'a {row["gender"]} is speaking')
            captions.append(f'the speaker is {row["gender"]}')
    
    return captions

def create_templates_fix(row: pd.Series, templates: typing.List[str]):
    captions = []
    if "pitch" in templates:
        if row["F0semitoneFrom27.5Hz_sma3nz_amean.quantile"] >= 0.7:
            captions.append("has a high pitch")
        elif row["F0semitoneFrom27.5Hz_sma3nz_amean.quantile"] <= 0.3:
            captions.append("has a low pitch")
        else:
            captions.append("has a normal pitch")

        if row["F0semitoneFrom27.5Hz_sma3nz_stddevNorm.quantile"] >= 0.7:
            captions.append("has a high pitch variation")
        elif row["F0semitoneFrom27.5Hz_sma3nz_stddevNorm.quantile"] <= 0.3:
            captions.append("has a low pitch variation")
        else:
            captions.append("has a normal pitch variation")

    if "intensity" in templates:
        if (row["equivalentSoundLevel_dBp.quantile"] >= 0.7) or (row["loudness_sma3_amean.quantile"] >= 0.7):
            captions.append("is loud")
            captions.append("has a high equivalent sound level")
        elif (row["equivalentSoundLevel_dBp.quantile"] <= 0.3) or (row["loudness_sma3_amean.quantile"] <= 0.3):
            captions.append("is almost silent")
        else:
            captions.append("loudness is just about right")

    if "duration" in templates:
        if row["Duration.quantile"] >= 0.7:
            captions.append("is long")
            captions.append("has a long duration")
            captions.append("is a long sentence")
            captions.append("lasts a long time")
            captions.append("has a big duration")
        elif row["Duration.quantile"] <= 0.3:
            captions.append("is short")
            captions.append("has a short duration")
            captions.append("is a short sentence")
            captions.append("lasts a little time")
            captions.append("has a small duration")
        else:
            captions.append("is of average length")
            captions.append("is of average duration")
            captions.append("is neither long or short")
            captions.append("duration is medium")
    
    if "jitter" in templates:
        if row["jitterLocal_sma3nz_amean.quantile"] >= 0.7:
            captions.append("has a high jitter")
        elif row["jitterLocal_sma3nz_amean.quantile"] <= 0.3:
            captions.append("has a low jitter")
        else:
            captions.append("has a normal jitter")

    if "shimmer" in templates:
        if row["shimmerLocaldB_sma3nz_amean.quantile"] >= 0.7:
            captions.append("has a high shimmer")
        elif row["shimmerLocaldB_sma3nz_amean.quantile"] <= 0.3:
            captions.append("has a low shimmer")
        else:
            captions.append("has a normal shimmer")

    if "emotion" in templates:
        if row["emotion"] == "no_agreement":
            captions.append("annotators disagree")
        else:
            captions.append(f'emotion is {row["emotion"]}')


    if "arousal" in templates:
        if row["arousal.quantile"] >= 0.7:
            captions.append("speaker is aroused")
            if row["emotion"] != "no_agreement":
                captions.append(f'speaker is very {row["emotion"]}')
        elif row["arousal.quantile"] <= 0.3:
            captions.append("speaker is calm")
            if row["emotion"] != "no_agreement":
                captions.append(f'speaker is not very {row["emotion"]}')
        else:
            captions.append("arousal is at an average level")

    if "dominance" in templates:
        if row["dominance.quantile"] >= 0.7:
            captions.append("speaker is dominant")
        elif row["dominance.quantile"] <= 0.3:
            captions.append("speaker has low dominance")
        else:
            captions.append("dominance is at an average level")

    if "valence" in templates:
        if row["valence.quantile"] >= 0.7:
            captions.append("has high valence")
        elif row["valence.quantile"] <= 0.3:
            captions.append("has low valence")
        else:
            captions.append("valence is at an average level")

    if "gender" in templates:
        if row["gender"] == row["gender"]:
            captions.append(f'a {row["gender"]} is speaking')
    return captions
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate templates")
    parser.add_argument("-dataset", "-d")
    parser.add_argument("-features", "-f")
    parser.add_argument("--dest", required=True)
    parser.add_argument(
        "--templates",
        default=[
            "emotion",
            "gender",
            "arousal",
            "valence",
            "dominance",
            "pitch",
            "shimmer",
            "jitter",
            "intensity",
            "duration"
        ],
        choices=[
            "emotion",
            "gender",
            "arousal",
            "valence",
            "dominance",
            "pitch",
            "shimmer",
            "jitter",
            "intensity",
            "duration"
        ],
        help="Variables to use for template creation"
    )
    parser.add_argument(
        "--filter-emotions",
        choices=[
            "no_agreement",
            "neutral",
            "fear",
            "sadness",
            "disgust",
            "happiness",
            "other",
            "anger",
            "contempt",
            "surprise"
        ],
        default=[
            "no_agreement",
            "neutral",
            "fear",
            "sadness",
            "disgust",
            "happiness",
            "other",
            "anger",
            "contempt",
            "surprise"
        ]
    )
    args = parser.parse_args()

    db = audformat.Database.load(args.dataset)
    df = db["categories.consensus.train"].df
    df = df.loc[df["emotion"].isin(args.filter_emotions)]
    df = pd.concat((df, db["dimensions.consensus.train"].get(index=df.index)), axis=1)
    df = pd.concat((df, db["speaker"].get(index=df.index)), axis=1)
    df["gender"] = df["speaker"].apply(lambda x: db.schemes["speaker"].labels[x]["gender"])
    features = pd.read_csv(args.features).set_index("file")

    feature_names = features.columns
    for feature in feature_names:
        features[f"{feature}.quantile"] = QuantileTransformer().fit_transform(features[feature].values.reshape(-1, 1))

    for dim in ["arousal", "valence", "dominance"]:
        df[f"{dim}.quantile"] = QuantileTransformer().fit_transform(df[dim].values.reshape(-1, 1))

    features = features.reindex(df.index)
    df = pd.concat((df, features), axis=1)

    os.makedirs(args.dest, exist_ok=True)
    for index, row in tqdm.tqdm(
        df.iterrows(), 
        total=len(df), 
        desc="Creating templates"
    ):
        templates = create_templates_fix(row, args.templates)
        with open(os.path.join(args.dest, f"{os.path.basename(index)}.json"), "w") as fp:
            json.dump(templates, fp, indent=2)