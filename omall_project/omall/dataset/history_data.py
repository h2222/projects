import os
import pandas as pd

def find_data_path(unzip=True):
    path_list=[]
    path = os.walk('.')
    next(path)
    for i, _, q in path:
        f = i+'/'+q[1]
        if unzip:
            os.system('gzip -d '+f)
            path_list.append(f[:-3])
        path_list.append(f)
    return path_list


def process_data(data_path, df):
    with open(data_path, 'r') as f:
        for i, l in enumerate(f.readlines()[:-1]):
            print(data_path,i)
            d = eval(str(l.strip())) # 获取one line data <dict>
            df = df.append(d, ignore_index=True)
    return df


def create_dataset(unzip=False):
    path_list = find_data_path(unzip)
    df = pd.DataFrame(columns=['user_id', 'ts', 'good_id', 'label'])
    for p in path_list:
        df = process_data(p, df)
    df = df.sort_values(by=['user_id'])
    return df


if __name__ == "__main__":
    df = create_dataset()
    print(df.shape)
    df.to_csv('/data/jiaxiang_data/collection_data-2020-7.csv', index=False, encoding='utf-8-sig')
