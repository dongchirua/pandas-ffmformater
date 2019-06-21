from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# make dataset
train, y = make_classification(n_samples=2000, n_features=5, n_informative=2, n_redundant=2, n_classes=2,
                               random_state=42)

train = pd.DataFrame(train, columns=['int1', 'int2', 'float1', 's1', 's2'])
train['id'] = [str(i) for i in range(1, len(train) + 1)]
train['int1'] = train['int1'].map(int) + np.random.randint(0, 8)
train['int2'] = train['int2'].map(int)
train['s1'] = np.log(abs(train['s1'] + 1)).round().map(str)
train['s2'] = np.log(abs(train['s2'] + 1)).round().map(str)
train['clicked'] = y


if __name__ == "__main__":
    print('Base data')
    print(train[0:10])

    # transform data
    categorical = ['int1', 'int2']
    numerical = ['float1', 's1', 's2']
    target = 'clicked'

    train_data, val_data = train_test_split(train, test_size=0.2)

    from FFMFormat import FFMformatter

    ffm_train = FFMformatter(categorical=categorical, numerical=numerical, label_column='clicked')
    data = ffm_train.fit_transform(train, train['clicked'].values)

    print('FFM data')
    print(data.head(10))
