import torch
import torch.nn as nn
import torch.nn.functional as F

from s3dataset import S3Checkpoint


def test_general_checkpointing(checkpoint_directory):
    tensor = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
    checkpoint_name = "general_checkpoint.pt"
    checkpoint = S3Checkpoint(region=checkpoint_directory.region)
    s3_uri = f"{checkpoint_directory.s3_uri}/{checkpoint_name}"
    with checkpoint.writer(s3_uri) as writer:
        torch.save(tensor, writer)

    loaded = torch.load(checkpoint.reader(s3_uri))

    assert torch.equal(tensor, loaded)


def test_nn_checkpointing(checkpoint_directory):
    nn_model = Net()
    checkpoint_name = "neural_network_model.pt"
    checkpoint = S3Checkpoint(region=checkpoint_directory.region)

    epoch = 5
    s3_uri = f"{checkpoint_directory.s3_uri}/{checkpoint_name}"
    loss = 0.4

    with checkpoint.writer(s3_uri) as writer:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": nn_model.state_dict(),
                "loss": loss,
            },
            writer,
        )

    loaded_nn_model = Net()

    # assert models are not equal before loading from checkpoint
    assert not nn_model.equals(loaded_nn_model)

    loaded_checkpoint = torch.load(checkpoint.reader(s3_uri))
    loaded_nn_model.load_state_dict(loaded_checkpoint["model_state_dict"])
    assert nn_model.equals(loaded_nn_model)

    loaded_epoch = loaded_checkpoint["epoch"]
    loaded_loss = loaded_checkpoint["loss"]
    assert loss == loaded_loss
    assert epoch == loaded_epoch

    # Assert that eval and train do not raise
    loaded_nn_model.eval()
    loaded_nn_model.train()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def equals(self, other_model: nn.Module) -> bool:
        for key_item_1, key_item_2 in zip(
            self.state_dict().items(), other_model.state_dict().items()
        ):
            if not torch.equal(key_item_1[1], key_item_2[1]):
                return False
        return True
