from DatasetAnalysisClass import CarPriceDatasetAnalysis


def test_dataset_analysis_class() -> None:
    """  Training a linear regression model for car price program,
    using gradient descent algorithm.
    The class CarPriceDatasetAnalysis instanciation loads data
    for car mileages and prices, provided by a data.csv file.
    Model training for the dataset will noramlize the dataset and
    instanciate class LinearRegressionGradientDescent.
    """
    test_model = CarPriceDatasetAnalysis()
    test_model.train_dataset(0.2, 500)
    return None


if __name__ == "__main__":
    """training model"""
    test_dataset_analysis_class()
