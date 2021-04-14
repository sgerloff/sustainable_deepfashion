<p align="center">
  <img src="https://github.com/sgerloff/sustainable-deepfashion-dash/blob/main/assets/sustainable_deepfashion_title.png?raw=true" width=512px/>
</p>

Circular fashion solves many ethical and environmental problems of the fast-fashion industry. However, finding items that suit your style from sustainable sources, such as the second-hand market, can prove to be quite hard. Our app demonstrates how we can solve this issue with the power of deep learning.

# [Prototype](https://sustainable-deepfashion-dash.herokuapp.com/) Solution

<p align="center">
  <a href="https://sustainable-deepfashion-dash.herokuapp.com/">
  <img src="https://github.com/sgerloff/sustainable_deepfashion/blob/main/docs/assets/dash-app-screenshot.jpg?raw=true" width=385px/>
  </a>
</p>

This app demonstrates our model has learned to understand the similarity between photos of fashion items with respect to their color, pattern, and more, instead of relying solely on the similarity between the images themselves.
On top of that, we allow users to shift their predictions to add stripes or floral patterns. 
While this functionality is limited by the rather small database (10000 items), it clearly demonstrates that this model can also be used for the inspiration and exploration of the user.
Currently, the database for the predictions is focused on short-sleeved tops, which the model was trained on.

[**Feel free to try it out for yourself!**](https://sustainable-deepfashion-dash.herokuapp.com/)<br/>
(Please be patient. Heroku is an amazing, free product, but may take some time to fire up the instance.)

# Concept
<p align="center">
  <img src="https://github.com/sgerloff/sustainable_deepfashion/blob/main/docs/assets/concept.jpg?raw=true" width=512px/>
</p>

Our app implements a **content-based recommendation** system, which works as follows. Given a query image from the user, the model is used to compute the representation vector for this input image. Next, the representation vector is compared against precomputed representation vectors stored in a database. From this database, the items with the smallest distance to the query image are chosen and returned to the user as suggestions. Optionally, the user can proceed to modify the computed representation, moving the suggestions into the desired direction, such as adding specific patterns. Here, we exploit the fact that the trained model outputs a well-behaved latent space.

# Data and Training

We teach the model similarities between fashion items, with respect to their colors, prints, and patterns. To this end, we employ the semi-hard triplet loss function, which operates on a batch containing both positive (matching items) and negative examples. The loss compares the hardest positive example, i.e. the positive pair with the largest distance, to a random negative example. Penalizing large distances between the positive pair while rewarding large distances between the negative pair, the model learns to place matching items close to each other in the latent space.

To train this model, we require labeled data for matching fashion items, which we obtain from three different sources. First, we utilize the [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) dataset, which contains commercial and user image pairs for various categories of clothes. Second, we have gathered pictures from friends and families, providing a high variety of different angles, zooms, and backgrounds for the same item. Finally, we have scraped data from [Vinted](https://www.vinted.de/), which directly reflects the distribution of data we want to operate on and provides a huge variety of different items. In addition, we augment the data by rotating and cropping the images randomly.

In total, we end up with a dataset containing up to 500.000 images of 10.000 different items from the category of short-sleeved tops, which we have focused on. The training was performed on K80 GPUs on both AWS and Google Cloud. To manage hyperparameter experimentation between multiple Data Scientists, we have implemented a custom setup, which executes a training run from a single instruction file.

## Data Cleaning
<p align="center">
  <img src="https://sgerloff.github.io/sharedplots/image_cleaner.gif" width=512px/>
</p>

As expected, the scraped data from Vinted needed to be cleaned before being used for training the model. To this end, we have developed a small GUI in PyQt5. This tool allowed us to quickly iterate through the set of images corresponding to a single item. During cleaning, we focused on removing pictures from different items, duplicate entries, tags, and any other ambiguous images.

# Top-k Accuracies (Validation)

<p align="center">
  <img src="https://github.com/sgerloff/sustainable_deepfashion/blob/main/docs/assets/test_train_split.jpg?raw=true" width=512px/>
</p>

Experimenting with various hyperparameters, such as the dimensions of the latent space, the distance metric (L2, angular distance, ...), data augmentation, as well as different architectures, we have constructed a test scenario to compare the different models. First, we split the data with respect to the number of images per item, where items with only two images are used for the test and validation and all others are used for training. We then split the items containing only a single pair of images into a test set and a validation set of equal size.

<p align="center">
  <img src="https://github.com/sgerloff/sustainable_deepfashion/blob/main/docs/assets/top_k_visualization.jpg?raw=true" width=512px/>
</p>

During testing and validating, the model computes the representation vectors for all items in the respective set. Choosing one item out of the set, we sort all others by their relative distance to that query item. From there we compute the accuracy to find the matching item in the Top-K (k=1,5,10) predictions. Since the dataset is composed of pairs of images for various items, there is exactly one right image to find. Thus circumventing the problem that an item with lots of examples in the dataset would be easier to find.
Finally, querying each item against all others, we compute the Top-K-Accuracy scores, which we use to compare different models.

# Results

<p align="center">
  <img src="https://github.com/sgerloff/sustainable_deepfashion/blob/main/docs/assets/topk_accuracies.png?raw=true" width=700px/>
</p>

Performing various different training runs, we find that our best performing model is a custom CNN with five convolutional layers followed by three dense layers. With a 20 dimensional latent space and angular similarity as a distance metric, we achieve 58% Top-10 accuracy. This exact model, is also employed in our [demo app](https://sustainable-deepfashion-dash.herokuapp.com/). This model performs better than both an unsupervised variational autoencoder, as well as models employing transfer learning. Furthermore, our custom architecture is much smaller in size, which allows for easy deployment and fast inference.

## Observations
### Variational Autoencoder (VAE)
Training a VAE can be quite challenging on its own, but the big advantage is the fact that it boils down to an unsupervised machine learning problem. Thus, we can use all images of fashion items that we have available. However, as expected, the VAE struggles to capture similarities between the items, instead of similarities of the image itself. For example, the VAE may group two images based on their distinct background if a user posts images from different items captured in a similar situation. As a result, the triplet loss models outperform the VAE with respect to matching items from different angles, zooms, and lighting conditions.

### Transfer Learning
For the transfer learning, we have employed various architectures (MobileNetV2, EfficientNetB0, ResNet50, VGG-16, ...) each pretrained on the imagenet dataset. However, we observed that smaller models perform better and in general these models tend to overfit already after the first epoch. In other words, the features extracted from these models are already too specific and are not useful to span a useful latent space. To counteract this, we tried to unfreeze some of the pretrained layers, however, this increased the training times tremendously.

# Authors

We are three Data Scientists that met at a Berlin Bootcamp called Data Science Retreat. This project was part of the Bootcamp and has been realized for the most part in 3-4 weeks. If you have feedback, suggestions, or want to chat, please contact us:
 
 - [Gert-Jan Dobbelaere](https://www.linkedin.com/in/gert-jan-dobbelaere/)
 - [Dr. Sascha Gerloff](https://www.linkedin.com/in/sascha-gerloff)
 - [Dr. Sergio Vechi](https://www.linkedin.com/in/sergiovechi/)

# Outlook and ToDo's

This project is currently on the level of proof of concept. In order to deploy our solution in a real app and actually improve the user experience in the second-hand market, we would like to work on various extensions:

 - Add more clothing categories, like pants, shoes, and more.
 - Check against a live database, capturing items currently available in the second-hand market.
 - Explore more powerful models, such as GAN.
 - Add options, to filter for size, color styles, and more.
