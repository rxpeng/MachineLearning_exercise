# MachineLearning_exercise

Just some simple exercises about element of marchine learning.

## [First exercise: Spam Message detection](./Spam_message_classfication)

Details: I use different approaches to establish relation between the text and the category, based on size of message, word count, special keywords, using term-frequency inverse document-frequency (tf-idf) transform. Namely,

1. Multinomial Naïve Bayes
2. DecisionTreeClassifier
3. AdaBoost
4. K Nearest Neighbors
5. Random Forest

Dataset Description: We use a database of 4000 text messages contains a collection of 538 spam messages, and a subset of 3,462  non-spam messages. The dataset is a csv file, in first column starts with the label of the message, followed by the second column which is the text message string. After preprocessing of the data and extraction of features, machine learning techniques are applied to the samples, and their performances are compared.



