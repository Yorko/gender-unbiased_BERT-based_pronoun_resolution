import pandas as pd


def scrape_url(url):
    """
    Get the title of the wikipedia page and replace "_" with white space
    """
    return url[29:].lower().replace("_", " ")


def check_name_in_string(name, string):
    """
    Check whether the name string is a substring of another string (i.e. wikipedia title)
    """

    return int(name.lower() in string)


def found_in_url(df):
    """

    Builds 2 simple binary features: 'A_in_URL' and 'B_in_URL'.

    :param df: pandas DataFrame with competition data
    :return: pandas DataFrame with 2 binary features
    """

    pred_df = pd.DataFrame(index=df.index)

    pred_df['A_in_URL'] = df.apply(lambda row: check_name_in_string(row["A"], scrape_url(row["URL"])), axis=1)
    pred_df['B_in_URL'] = df.apply(lambda row: check_name_in_string(row["B"], scrape_url(row["URL"])), axis=1)

    return pred_df

