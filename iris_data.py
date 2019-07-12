import pandas as pd
import tensorflow as tf

TRAIN_FILE='iris_training.csv'
TEST_FILE='iris_test.csv'

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def load_data(y_name='Species'):
    """returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    # header=0 to replace existing names
    train = pd.read_csv(TRAIN_FILE, names=CSV_COLUMN_NAMES, header=0)
    # Dataframe.pop first execute, so the train has only four columns left
    train_x, train_y = train, train.pop(y_name)

    # header=0 to replace existing names
    test = pd.read_csv(TEST_FILE, names=CSV_COLUMN_NAMES, header=0)
    # Dataframe.pop first execute, so the train has only four columns left
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    # note: pandas.Dataframe is different with tf.dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    # note: repeat to make the input pipeline always has data
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset
    return dataset

