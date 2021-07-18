""" Our model — Neural Collaborative Filtering (NCF) """


"""
User Embeddings

Before we dive into the architecture of the model, let’s familiarize ourselves with the concept of embeddings. 
An embedding is a low-dimensional space that captures the relationship of vectors from a higher dimensional space.


Of course, we are not restricted to using just 2 dimensions to represent our users. We can use an arbitrary number of 
dimensions to represent our users. A larger number of dimensions would allow us to capture the traits of each user
 more accurately, at the cost of model complexity. In our code, we’ll use 8 dimensions (which we will see later).
"""

# -------------------------------------------------------------------------------------------------

"""
Model Architecture
Now that we have a better understanding of embeddings, we are ready to define the model architecture. 
As you’ll see, the user and item embeddings are key to the model.
"""

# -------------------------------------------------------------------------------------------------

""" Now, let’s define this NCF model using PyTorch Lightning! """

import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)

        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            all_movieIds (list): List containing all movieIds (train + test)
    """

    def __init__(self, num_users, num_items, ratings, all_movieIds):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_movieIds = all_movieIds

    def forward(self, user_input, item_input):
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MovieLensTrainDataset(self.ratings, self.all_movieIds),
                          batch_size=512, num_workers=4)


"""
Note: One advantage of PyTorch Lightning over vanilla PyTorch is that you don’t need to write your own boiler plate
 training code. Notice how the Trainer class allows us to train our model with just a few lines of code.
"""

num_users = ratings['userId'].max()+1
num_items = ratings['movieId'].max()+1
all_movieIds = ratings['movieId'].unique()

model = NCF(num_users, num_items, train_ratings, all_movieIds)

trainer = pl.Trainer(max_epochs=5, gpus=1, reload_dataloaders_every_epoch=True,
                     progress_bar_refresh_rate=50, logger=False, checkpoint_callback=False)

trainer.fit(model)