<p align="center">
  <img src="https://github.com/sgerloff/sustainable-deepfashion-dash/blob/main/assets/sustainable_deepfashion_title.png?raw=true" width=512px/>
</p>

Circular fashion solves many ethical and environment problems of the fast-fashion industry. However, finding items that suit your style from sustainable sources, such as the second-hand market, can prove to be quite hard. Our app demonstrates how we can solve this issue with the power of deep-learning.

# [Prototype](https://sustainable-deepfashion-dash.herokuapp.com/) Solution

<p align="center">
  <a href="https://sustainable-deepfashion-dash.herokuapp.com/">
  <img src="https://github.com/sgerloff/sustainable_deepfashion/blob/main/docs/assets/dash-app-screenshot.jpg?raw=true" width=385px/>
  </a>
</p>

This app demonstrates our model has learned to understand the similarity between photos of fashion items with respect to their color, pattern and more, instead of relying soley on the similarity between the images themselves.
On top, we allow users to shift their predictions to add stripes or floral patterns. 
While, this functionality is limited by the rather small database (10000 items), it clearly demonstrates that this model can also be used for inspiration and exploration of the user.
Currently, the database for the predictions is focused on short-sleeved tops, which the model was trained on.

[**Feel free to try it out for yourself!**](https://sustainable-deepfashion-dash.herokuapp.com/)

# Concept

![Concept](docs/assets/concept.png)

# Top-k accuracies

![Accuracies](docs/assets/topk_accuracies.png)

