import tempfile
from spam_classifier.data import download_dataset, load_dataset
from spam_classifier.preprocess import prepare_features


def test_data_download_and_prepare():
    csv = download_dataset(dest=tempfile.gettempdir() + "/sms_spam_no_header.csv")
    df = load_dataset(csv)
    splits, vec = prepare_features(df, max_features=100)
    assert "X_train" in splits
    assert vec is not None
