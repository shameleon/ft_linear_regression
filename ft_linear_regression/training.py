from DatasetAnalysisClass import CarPriceDatasetAnalysis

def test_dataset_analysis_class() -> None:
    """ """
    test_model = CarPriceDatasetAnalysis()
    test_model.train_dataset(0.2, 500)
    return None


if __name__ == "__main__":
    """training model"""
    test_dataset_analysis_class()