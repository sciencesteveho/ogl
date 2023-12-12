import pytest
from unittest.mock import patch, MagicMock
from gnn import create_model, train, test, inference, main


@pytest.fixture
def mock_data_loader():
    # Implement a mock data loader for testing
    # For simplicity, let's assume a basic iterable data loader
    return [(mock_data(), mock_targets()), (mock_data(), mock_targets())]


def mock_data():
    # Implement a function to generate mock input data
    pass


def mock_targets():
    # Implement a function to generate mock target values
    pass


def test_create_model():
    # Test create_model function with different model types
    model_types = ["GraphSAGE", "GCN", "GATv2", "MLP", "GPS"]

    for model_type in model_types:
        model = create_model(model_type, in_size=32, embedding_size=64, out_channels=128, num_layers=3)
        assert model is not None
        assert isinstance(model, YourModelClass)  # Replace YourModelClass with the actual model class
        # Add more specific assertions based on your model structure and properties


def test_train(mock_data_loader):
    # Mocking torch.cuda.is_available() to always return False for testing
    with patch('torch.cuda.is_available', return_value=False):
        device = "cpu"
        model = MagicMock()
        optimizer = MagicMock()
        epoch = 1

        loss = train(model, device, optimizer, mock_data_loader, epoch, gps=False)
        assert isinstance(loss, float)
        # Add more assertions based on your specific requirements


def test_test(mock_data_loader):
    # Mocking torch.cuda.is_available() to always return False for testing
    with patch('torch.cuda.is_available', return_value=False):
        device = "cpu"
        model = MagicMock()
        epoch = 1
        mask = "val"

        accuracy = test(model, device, mock_data_loader, epoch, mask, gps=False)
        assert isinstance(accuracy, float)
        # Add more assertions based on your specific requirements


def test_inference(mock_data_loader):
    # Mocking torch.cuda.is_available() to always return False for testing
    with patch('torch.cuda.is_available', return_value=False):
        device = "cpu"
        model = MagicMock()
        epoch = 1

        rmse, outs, labels = inference(model, device, mock_data_loader, epoch, gps=False)
        assert isinstance(rmse, float)
        assert isinstance(outs, list)
        assert isinstance(labels, list)
        # Add more assertions based on your specific requirements


def test_main():
    # Mocking argparse.ArgumentParser to parse arguments from a list
    with patch('argparse.ArgumentParser.parse_args',
               return_value=MagicMock(experiment_config="config.yaml",
                                      model="GCN",
                                      layers=2,
                                      dimensions=600,
                                      seed=42,
                                      loader="neighbor",
                                      batch_size=1024,
                                      learning_rate=1e-4,
                                      idx="true",
                                      device=0,
                                      graph_type="full",
                                      zero_nodes="false",
                                      randomize_node_feats="false",
                                      early_stop="true",
                                      expression_only="false",
                                      randomize_edges="false")):
        main()
        # Add more assertions based on your specific requirements
