import pandas as pd
import cornac


def gen_cornac_dataset(data_path, t=-1) -> object:
    df = pd.read_csv(data_path)
    df = df[df.rating > t]
    data = cornac.data.Dataset.from_uir(df.itertuples(index=False))
    return df, data
