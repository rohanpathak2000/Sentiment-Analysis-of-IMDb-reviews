getwd()
setwd("C:/FallSem 21-22/CSE3505 - Data Analytics/Project Prep/Sentiment Analysis of IMDb reviews")
data <- read.csv("movie_reviews.csv")
#head(data)
#View(data)
names(data)
pacman::p_load(tm)
pacman::p_load(tmap)
#pacman::p_load(SnowballC)

str(data)
#corpus <- iconv(df$review, to="utf-8")
corpus <- Corpus(VectorSource(data$review))
inspect(corpus[1:5])
corpus

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
#inspect(cleanset[1:2])

#cleanset <- tm_map(cleanset, removeWords, c("scene","watch"))


removeBRs <- function(x) gsub('<.*?>', '', x)
cleanset <- tm_map(cleanset, content_transformer(removeBRs))

#Stemming
cleanset <- tm_map(cleanset,stemDocument)
#inspect(cleanset[1:2])
####!!!! Trivial cleaning !!! ####

######  Extended Cleaning ########
frequencies <- DocumentTermMatrix(cleanset)

#inspect(frequencies[1:20,1:20])


findFreqTerms(frequencies, lowfreq = 10)

#Remove sparse terms
#Only contains terms which are present in >=2% of the reviews
sparse <- removeSparseTerms(frequencies, 0.98)
#frequencies
#sparse

final_reviews <- as.data.frame(as.matrix((sparse)))
colnames(final_reviews) <- make.names(colnames(final_reviews))

cols <- names(final_reviews)
cols

#for (col in cols) {
#print(sum(final_reviews[col]))
#}

for (col in cols) {
  if(sum(final_reviews[col]) < 2000) {
    final_reviews[col] <- NULL 
  }
}

dim(final_reviews)
final_reviews$sentiment <- data$sentiment

#install.packages("caret")
pacman::p_load(e1071)
pacman::p_load(caret)


final_reviews$sentiment <- ifelse(final_reviews$sentiment == "negative",0,1)
final_reviews$sentiment <- factor(final_reviews$sentiment)

#Using 50,000 sentiments 
sample_data <- final_reviews

install.packages('caTools')
pacman::p_load(caTools)

set.seed(123)
split = sample.split(sample_data$sentiment, SplitRatio = 0.80)

training_set = subset(sample_data, split == TRUE)
testing_data = subset(sample_data, split == FALSE)
dim(training_set)
pacman::p_load(e1071)
View(training_set)

