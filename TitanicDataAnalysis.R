
# Predictive model for Titanic passenger survival -  data from Kaggle #
# Based on David Langer's excellent intro videos #

# Load raw data
Titanic.train <- read.csv("Titanic/train.csv", header = TRUE)
Titanic.test <- read.csv("Titanic/test.csv", header = TRUE)


# Review data
str(Titanic.train)


# Fix classes
Titanic.train$survived <- as.factor(Titanic.train$survived)
Titanic.train$pclass <- as.factor(Titanic.train$pclass)

# Add variables
Titanic.train$notAlone <- as.factor(Titanic.train$sibsp + Titanic.train$parch)

# Data survival rate
table(Titanic.train$survived)

# Checking for missing values - better not to use the age variable
missing <- colSums(is.na(Titanic.train))

#==============================================================================

# Visualize relationship of variables with survived
library(ggplot2)

# high death count in 3rd class
ggplot(Titanic.train) +
  geom_bar(aes(x = pclass, fill = survived)) +
  xlab("Pclass") +
  ylab("Total Count") +
  labs(fill = "Survived") 

# high death count for males, low for females
ggplot(Titanic.train) +
  geom_bar(aes(x = sex, fill = survived)) +
  xlab("sex") +
  ylab("Total Count") +
  labs(fill = "Survived") 

# high death count for lone travelers and big groups (families) with 5+ people
ggplot(Titanic.train) +
  geom_bar(aes(x = notAlone, fill = survived)) +
  xlab("not traveling alone") +
  ylab("Total Count") +
  labs(fill = "Survived") 

# high death count for cheaper tickets
ggplot(Titanic.train) +
  geom_histogram(aes(x = fare, fill = survived),binwidth = 10) +
  xlab("fare") +
  ylab("Total Count") +
  labs(fill = "Survived") 

# high death count for cheaper tickets and at 3rd class
ggplot(Titanic.train) +
  geom_histogram(aes(x = fare, fill = survived),binwidth = 10) +
  facet_wrap(~pclass) + 
  ggtitle("Pclass") +
  xlab("fare") +
  ylab("Total Count") +
  labs(fill = "Survived") 

# high death count for embarked at Southampton
ggplot(Titanic.train) +
  geom_bar(aes(x = embarked, fill = survived)) +
  xlab("embarked") +
  ylab("Total Count") +
  labs(fill = "Survived") 



# retrieve title from name variable as a replacement to sex and age variables
library(stringr)

title.split <- sapply(str_split(Titanic.train$name, ", "), "[", 2)
title <- sapply(str_split(title.split, " "), "[", 1)
unique(title)
Titanic.train$title <- as.factor(title)

# Merge titles
Titanic.train$title[title %in% c("Lady.", "the", "Mme.")] <- "Mrs."
Titanic.train$title[title %in% c("Ms.", "Mlle.")] <- "Miss."
Titanic.train$title[title %in% c("Sir.", "Jonkheer.", "Don.", "Col.", "Capt.", "Major.", "Rev.", "Dr.")] <- "Mr."
table(Titanic.train$title)



# high death count for adult males
ggplot(Titanic.train) +
  geom_bar(aes(x = title, fill = survived)) +
  xlab("title") +
  ylab("Total Count") +
  labs(fill = "Survived") 

# Variables that look meaningful - pclass, title, fare, notAlone

#==============================================================================

# Another way to assess importance of features in using Mutual Information
library(infotheo)

mutinformation(Titanic.train$survived, Titanic.train$pclass)
mutinformation(Titanic.train$survived, Titanic.train$title)
mutinformation(Titanic.train$survived, Titanic.train$notAlone)
mutinformation(Titanic.train$survived, Titanic.train$embarked)
mutinformation(Titanic.train$survived, Titanic.train$sex)

# for numeric variables
mutinformation(Titanic.train$survived, discretize(Titanic.train$fare))


#==============================================================================

## Test for a good predictive model ##

library(caret)
set.seed(3333)

# Create Create Stratified folds
cv.10.folds <- createMultiFolds(Titanic.train$survived, k = 10, times = 10)

# Check ratio
table(Titanic.train$survived[cv.10.folds[[33]]])
table(Titanic.train$survived)


#==================== Models ====================
## Logistic Regression ##
set.seed(2222)

# Train model with all meaningful variables
logist.train.1 <- Titanic.train[, c("pclass", "title", "fare", "notAlone")]

ctrl <- trainControl(method = "repeatedcv", number = 5 , repeats = 10,
                         index = cv.10.folds)

ctrl.roc <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                       index = cv.10.folds, classProbs = TRUE, 
                       summaryFunction = twoClassSummary)


logist.1 <- train(logist.train.1,
                  y = Titanic.train$survived,
                  method = "glm",
                  trControl = ctrl)

logist.1


# Train model without pclass, probably correlated with fare
logist.train.2 <- Titanic.train[, c("title", "fare", "notAlone")]

logist.2 <- train(logist.train.2,
                  y = Titanic.train$survived,
                  method = "glm",
                  family = binomial,
                  metric  ="ROC",
                  trControl = ctrl.roc)

logist.2

# Train model without title
logist.train.3 <- Titanic.train[, c("pclass", "fare", "notAlone")]

logist.3 <- train(logist.train.3,
                  y = Titanic.train$survived,
                  method = "glm",
                  trControl = ctrl)

logist.3



#==============================================================


## Random Forest ##
set.seed(2222)

# Train model with all meaningful variables
rf.train.1 <- Titanic.train[, c("pclass", "title", "fare", "notAlone")]

ctrl.rf <- trainControl(method = "repeatedcv", number = 5, repeats = 10,
                        index = cv.10.folds)


rf.1 <- train(x = rf.train.1, y = Titanic.train$survived, 
              method = "rf",
              ntree = 100, 
              trControl = ctrl.rf)


rf.1



# Train model without pclass, probably correlated with fare
rf.train.2 <- Titanic.train[, c("title", "fare", "notAlone")]

set.seed(2222)
rf.2 <- train(x = rf.train.2, y = Titanic.train$survived, 
              method = "rf",
              ntree = 100, 
              trControl = ctrl.rf)


rf.2




# Train model without fare
rf.train.3 <- Titanic.train[, c("pclass", "title", "notAlone")]

set.seed(2222)
rf.3 <- train(x = rf.train.3, y = Titanic.train$survived, 
              method = "rf",
              ntree = 100, 
              trControl = ctrl.rf)


rf.3



# Train model without notAlone
rf.train.4 <- Titanic.train[, c("pclass", "title", "fare")]

set.seed(2222)
rf.4 <- train(x = rf.train.4, y = Titanic.train$survived, 
              method = "rf",
              ntree = 100, 
              trControl = ctrl.rf)


rf.4


# Train model only fare and title
rf.train.5 <- Titanic.train[, c("title", "fare")]

set.seed(2222)
rf.5 <- train(x = rf.train.5, y = Titanic.train$survived, 
              method = "rf",
              ntree = 100, 
              trControl = ctrl.rf)


rf.5


# Best results are:
# and model of Random Forest - "pclass", "title", "fare", accuracy of 0.8362
# for the Random Forest full model. accuracy of 0.8360

#==============================================================================

# Submission to Kaggle

# Add and edit variables in the test set

# Fix classes
Titanic.test$survived <- factor(Titanic.test$survived, levels = c(0,1), labels = c("died","survived"))
Titanic.test$pclass <- as.factor(Titanic.test$pclass)

# Add variables
Titanic.test$notAlone <- as.factor(Titanic.test$sibsp + Titanic.test$parch)

title.split <- sapply(str_split(Titanic.test$name, ", "), "[", 2)
title <- sapply(str_split(title.split, " "), "[", 1)
unique(title)
Titanic.test$title <- as.factor(title)

# Merge titles
Titanic.test$title[title %in% c("Lady.", "the", "Mme.", "Dona.")] <- "Mrs."
Titanic.test$title[title %in% c("Ms.", "Mlle.")] <- "Miss."
Titanic.test$title[title %in% c("Sir.", "Jonkheer.", "Don.", "Col.", "Capt.", "Major.", "Rev.", "Dr.")] <- "Mr."

# Final test set
Final.test <- Titanic.test[, c("pclass", "title", "fare", "notAlone")]


# Make predictions
preds <- predict(rf.4, Final.test)

# missing values error
missing <- colSums(is.na(Final.test))
# fix one missing value in fare
Final.test$fare[which(is.na(Final.test$fare))] <- 0
# re-leveling the factor of title
Final.test$title <- factor(Final.test$title, levels = levels(Titanic.train$title))

# Make predictions - take 2
preds <- predict(rf.4, Final.test)
table(preds)

# Write out a CSV file for submission to Kaggle
submit.df <- data.frame(PassengerId = rep(892:1309), Survived = preds)

write.csv(submit.df, file = "Titanic_2.csv", row.names = FALSE)


