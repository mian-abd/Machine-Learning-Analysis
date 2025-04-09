"IMDB dataset", contains a set of 50,000 highly-polarized reviews from the Internet Movie Database.
They are split into 25,000 reviews for training and 25,000 reviews for testing, each set consisting in 50% negative 
and 50% positive reviews.

It has already been preprocessed:

First, each review (sequences of words) has been turned into sequences of integers, where each integer stands 
for a specific word in a dictionary. In our case, we only kept the top 1000 most frequently 
occurring words in the training data. Rare words were discarded. 
This allows us to work with vector data of manageable size.

Each review was then, one-hot-encoded to turn it into vectors of 0s and 1s. 
Concretely, this would mean for instance turning the sequence [3, 5] into a 1000-dimensional 
vector that would be all-zeros except for indices 3 and 5, which would be ones.

- The variables X_train and X_test are both matrices of 25000x1000 dimension. (25K: number of reviews,
1000: size of the vocabulary). They contain the final preprocessed reviews, each review being a list of 
one-hot encoded word indices (encoding a sequence of words). 

- The variables y_train and y_test are lists of 0s and 1s, 
where 0 stands for "negative" and 1 stands for "positive" review.
