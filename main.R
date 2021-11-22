rm(list=ls())
setwd("C:/FallSem 21-22/CSE3505 - Data Analytics/Project Prep/Sentiment Analysis of IMDb reviews")
data <- read.csv("movie_reviews.csv")
#head(data)
#View(data)
pacman::p_load(tm)
pacman::p_load(tmap)
pacman::p_load(SnowballC)

# str(data)
# corpus <- iconv(df$review, to="utf-8")
# Converting the reviews into a corpus in which each review can be treated as a document. 
corpus <- Corpus(VectorSource(data$review))
#inspect(corpus[1:5])
#corpus

# ---- Data Cleaning ------#

#Lower case
corpus <- tm_map(corpus,tolower)
#Removing Numbers
corpus <- tm_map(corpus, removeNumbers)
#Removing punctuation
corpus <- tm_map(corpus, removePunctuation)
#Stripping white space
cleanset <- tm_map(corpus, stripWhitespace)

#Removing stop words
cleanset <- tm_map(cleanset, removeWords, stopwords("english"))

#Removing html tags
removeTags <- function(x) gsub('<.*?>', '', x)
cleanset <- tm_map(cleanset, content_transformer(removeTags))
wordsToRemove <- c("movie", "film", "one","make", "scene", "show","get","see","stories","watch","even","realli", "will", "character", "look","just","time", "made", "people", "first","thing","br","awaybr")
cleanset <- tm_map(cleanset, removeWords,wordsToRemove)

# Stemming
cleanset <- tm_map(cleanset,stemDocument)

# Creates a document matrix which is a Bag-of-Words model with Term Frequency
frequencies <- DocumentTermMatrix(cleanset)

#head(df)
#inspect(frequencies[1:20,1:20])

#Remove sparse terms
#Only contains terms which are present in >=5% of the reviews
sparse <- removeSparseTerms(frequencies, 0.95)

#Converting the matrix into a dataframe
final_reviews <- as.data.frame(as.matrix((sparse)))
colnames(final_reviews) <- make.names(colnames(final_reviews))


cols <- names(final_reviews)

# Removing words which have a frequency less than 2000 in all the reviews 
for (col in cols) {
  if(sum(final_reviews[col]) < 2000) {
     final_reviews[col] <- NULL 
  }
}

final_reviews$sentiment <- data$sentiment

pacman::p_load(e1071)
pacman::p_load(caret)


# Label encoding the categorical data
final_reviews$sentiment <- ifelse(final_reviews$sentiment == "negative",0,1)
#final_reviews$sentiment <- factor(final_reviews$sentiment)
 

#Analysis
frequency <- colSums(as.matrix(final_reviews[,-dim(final_reviews)[2]]))                                  
freq <- sort(frequency, decreasing = TRUE)
words <- names(freq)
pacman::p_load(wordcloud)
wordcloud(words[1:100], freq[1:100],scale=c(1.5,.5), colors = c("red","green","blue"))
pacman::p_load(ggplot2)
barplot(freq[1:10],
        main = "Most frequently used 10 words",
        xlab = "Words",
        ylab = "Frequency",
        names.arg = words[1:10],
        col = "darkred"
)

pacman::p_load(caTools)

set.seed(123)

# Splitting the data into training and testing 
split = sample.split(final_reviews$sentiment, SplitRatio = 0.8)
training_set = subset(final_reviews, split == TRUE)
testing_data = subset(final_reviews, split == FALSE)
dims <- dim(training_set)
features_train <- training_set[,-dims[2]]
features_test <- testing_data[,-dims[2]]
labels_train <- training_set[,dims[2]]
labels_test <- testing_data[,dims[2]]

features_train_matrix <- data.matrix(features_train)
features_test_matrix <- data.matrix(features_test)
labels_train_matrix <- data.matrix(labels_train)

pacman::p_load(Metrics)
### Naive Bayes Classification ###
clf <- naiveBayes(sentiment ~ .,  data = training_set, laplace = 1)
#Predictions
y_pred <- predict(clf, features_test)
#Evaluation Metrics
precision <- posPredValue(as.factor(y_pred), as.factor(testing_data$sentiment), positive="1")
recall <- sensitivity(as.factor(y_pred), as.factor(testing_data$sentiment), positive="1")
Accuracy <- accuracy(testing_data$sentiment,y_pred)
Accuracy
precision
recall


# Accuracy - 78.98%
# Precision - 77.5%
# Recall - 81.68%


pacman::p_load(ROCR)
pacman::p_load(glmnet)
pacman::p_load(dplyr)

#Logistic Regression with Lasso Regularization
cv.lasso <- cv.glmnet(features_train_matrix, labels_train_matrix, alpha = 1, family = "binomial")
model <- glmnet(features_train_matrix, labels_train_matrix, alpha = 1, family = "binomial",
                lambda = cv.lasso$lambda.min)
probabilities <- model %>% predict(newx = features_test_matrix)
labels_pred <- ifelse(probabilities > 0.5, 1, 0)
#Evaluation Metrics
Accuracy <- accuracy(labels_test,labels_pred)
precision <- posPredValue(as.factor(labels_pred), as.factor(labels_test), positive="1")
recall <- sensitivity(as.factor(labels_pred), as.factor(labels_test), positive="1")
Accuracy
precision
recall

# Accuracy - 80.99%
# Precision - 86.02%
# Recall - 74%

#Logistic Regression with Ridge Regularization
cv.ridge <- cv.glmnet(features_train_matrix, labels_train_matrix, alpha = 0, family = "binomial")
model <- glmnet(features_train_matrix, labels_train_matrix, alpha = 0, family = "binomial",
                lambda = cv.ridge$lambda.min)
probabilities <- model %>% predict(newx = features_test_matrix)
labels_pred <- ifelse(probabilities > 0.5, 1, 0)
#Evaluation Metrics
Accuracy <- accuracy(labels_test,labels_pred)
precision <- posPredValue(as.factor(labels_pred), as.factor(labels_test), positive="1")
recall <- sensitivity(as.factor(labels_pred), as.factor(labels_test), positive="1")
Accuracy
precision
recall

# Accuracy - 80.7%
# Precision - 86.46%
# Recall - 72.8%


## Logistic model without regularization
logistic_model <- glm(sentiment ~ ., 
                      data = training_set, 
                      family = "binomial")
predict_reg <- predict(logistic_model, 
                       testing_data, type = "response")
predict_reg <- ifelse(predict_reg >0.5, 1, 0)
#Evaluation Metrics
Accuracy <- accuracy(labels_test,predict_reg)
precision <- posPredValue(as.factor(predict_reg), as.factor(labels_test), positive="1")
recall <- sensitivity(as.factor(predict_reg), as.factor(labels_test), positive="1")
Accuracy
precision
recall

# Accuracy - 82.16%
# Precision - 81.25%
# Recall - 83.6%


## Random Forest Classifier
pacman::p_load(randomForest)
rf <- randomForest(
  sentiment ~ .,
  data=training_set,
  ntree = 200,
  sampsize = 0.80*nrow(training_set)
)
#print(rf)
y_pred = predict(rf, newdata = features_test)
Accuracy <- accuracy(labels_test,y_pred)
precision <- posPredValue(as.factor(y_pred), as.factor(labels_test), positive="1")
recall <- sensitivity(as.factor(y_pred), as.factor(labels_test), positive="1")
Accuracy
precision
recall

# Accuracy - 80.46%
# Precision - 80%
# Recall - 81.38%

## Neural network using built in package
pacman::p_load(neuralnet)
n <- neuralnet(
  sentiment ~ .,
  data = training_set,
  hidden = 1,
  err.fct = "ce",
  linear.output = FALSE
)
output <- neuralnet::compute(n, features_test)
p1 <- output$net.result
pred <- ifelse(p1>0.5, 1, 0)
pred
Accuracy <- accuracy(labels_test, pred)
precision <- posPredValue(as.factor(pred), as.factor(testing_data$sentiment), positive="1")
recall <- sensitivity(as.factor(pred), as.factor(testing_data$sentiment), positive="1")
Accuracy
precision
recall



tab1 <- table(pred, testing_data$sentiment)
tab1

# Accuracy - 82.34%
# Recall - 83.74%
# Precision - 81.45%              

pacman::p_load(rpart)


# Decision Tree 
training_set$sentiment <- factor(training_set$sentiment)
tree <- rpart(sentiment ~., data = training_set)
testing_data$sentiment <- factor(testing_data$sentiment)
pred <- predict(tree, testing_data, type="class")
Accuracy <- accuracy(labels_test, pred)
precision <- posPredValue(as.factor(pred), as.factor(labels_test), positive="1")
recall <- sensitivity(as.factor(pred), as.factor(labels_test), positive="1")
Accuracy
precision
recall

# Accuracy - 71.03%
# Precision - 67.13%
# Recall - 82.38%

