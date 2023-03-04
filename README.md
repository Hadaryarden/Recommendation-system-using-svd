###### What is a recommendation system?

A recommendation system is an artificial intelligence or AI algorithm, usually associated with machine learning, that uses Big Data to suggest or recommend additional products to consumers

here I used svd (singular value decomposition) to solve this problem.

Building a recommendation system based on artists that are on Spotify. 
The data contains - 2000 users and close to 20,000 artists.
We chose to work with svd because we assumed that this implementation could return us the most accurate answer. We chose K according to the elbow method.
First we split artist_user into a training set and a test set. We used a matrix calculation and then calculated the RMSE for the prediction we created for a different number of K features, to see how to choose the optimal K.
After minimizing the RMSE for the user-artist file we divided, and getting a good enough result, we returned to the original problem. artist_user was used as the training set and the test file was used as the test file. in the for loop
We went through each line in the test file and found the prediction we got from the svd. In the svd function we created a matrix
Predictions therefore, for each artist and each stranger, if they exist in the training set, we accessed the corresponding row and column.
In the test file we received there is one user who does not appear in the user-artist file. That means this user heard
So far only one artist and we are trying to estimate this number of plays. We decided to refer to the user
This is as a new user. And so we decided to look for users in the user-artist file that we could refer to
Even as new users, we saw that there are eight users in the user-artist file who heard one artist
only. However, we also saw that among these users there are 2 users with a very high number of plays, which can bias the predicted number of plays if we use the average. That's why we chose to calculate their median and use this value as the new user's prediction in the test file.
