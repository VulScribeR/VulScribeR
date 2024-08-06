import yaml
import pandas as pd

# Specify the path to your YAML file
yaml_file_path = './keys.yml'


def get_chatgpt_token():
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
        return data.get('openai', {}).get('chatgpt')


def print_df(df, str_fn= lambda x: x):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):
        print(str_fn(df.to_string()))

def print_df_code(df):
    print_df(df, str_fn= lambda x: str(x).replace("\\n","\n"))
