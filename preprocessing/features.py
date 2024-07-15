import argparse
import audformat
import opensmile
import os
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate templates")
    parser.add_argument("-dataset", "-d")
    parser.add_argument("-features", "-f")
    args = parser.parse_args()

    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, verbose=True, num_workers=8)

    db = audformat.Database.load(os.path.join(args.dataset, "converted"))
    features = smile.process_index(db.files, root=os.path.join(args.dataset, "original"))
    features = features.reset_index()
    features["Duration"] = (features["end"] - features["start"]).dt.total_seconds()
    features = features.drop(["start", "end"], axis=1)
    os.makedirs(os.path.dirname(args.features), exist_ok=True)
    features.to_csv(args.features, index=False)
