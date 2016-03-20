#--------------------------------------------------------------------------------------
# survival ... Survival (0 = No; 1 = Yes)
# pclass ..... Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# name ....... Name
# sex ........ Sex
# age ........ Age
# sibsp ...... Number of Siblings/Spouses Aboard
# parch ...... Number of Parents/Children Aboard
# ticket ..... Ticket Number
# fare ....... Passenger Fare
# cabin ...... Cabin
# embarked ... Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
#--------------------------------------------------------------------------------------

# Set working directory and import datafiles
setwd("~/projects/r/titanic")
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

# Install and load required packages for decision trees and forests
library(rpart)
#install.packages('randomForest')
library(randomForest)
#install.packages('party')
library(party)


## data clean-up
# Cast target attribute to factor
train$Survived <- as.factor(train$Survived)
levels(train$Survived) <- c('Perished', 'Survived')

# Join together the test and train sets for easier feature engineering
test$Survived <- NA
combined <- rbind(train, test)


# Convert Name feature to a string
combined$Name <- as.character(combined$Name)

# Engineered variable: Title
combined$Title <- sapply(combined$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combined$Title <- sub(' ', '', combined$Title)
# Combine small title groups
combined$Title[combined$Title %in% c('Mme', 'Mlle')] <- 'Ms'
combined$Title[combined$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combined$Title[combined$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
# Convert to a factor
combined$Title <- factor(combined$Title)

# Engineered variable: Family size
combined$FamilySize <- combined$SibSp + combined$Parch + 1

# Engineered variable: Family
#-------------------------------------------------------------------
# not happy with this whole FamilyID thing.  Needs improvement
# include the hyphen, to try to improve family grouping
combined$Surname <- sapply(combined$Name, FUN=function(x) {strsplit(x, split='[,.-]')[[1]][1]})
combined$FamilyID <- paste(as.character(combined$FamilySize), combined$Surname, sep=":")
combined$FamilyID[combined$FamilySize <= 2] <- 'Small'
# Delete erroneous family IDs
famIDs <- data.frame(table(combined$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 3,]
combined$FamilyID[combined$FamilyID %in% famIDs$Var1] <- 'Small'
# Convert to a factor
combined$FamilyID <- factor(combined$FamilyID)
#-------------------------------------------------------------------

# Fill in Age NAs
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize, data=combined[!is.na(combined$Age),], method="anova")
combined$Age[is.na(combined$Age)] <- predict(Agefit, combined[is.na(combined$Age),])

# Fill in Embarked blanks
train$Embarked <- sub("^$","S",train$Embarked)
combined$Embarked <- factor(combined$Embarked)

# Fill in Fare NAs
# some fares are 0.00, which is probably the same as NA
combined$Fare[combined$Fare == 0] <- NA
# A predictive model for missing fare way better than median value
Farefit <- rpart(Fare ~ Pclass + Sex + SibSp + Parch + Embarked + Title + FamilySize, data=combined[!is.na(combined$Fare),], method="anova")
combined$Fare[is.na(combined$Fare)] <- predict(Farefit, combined[is.na(combined$Fare),])

# Split back into test and train sets
split <- nrow(train)
last <- nrow(combined) -1
train <- combined[1:split, ]
test <- combined[(split+1):last, ]


## build a new test data set, with survival data, so that we can test the model prior to submitting to Kaggle.
# randomly choose 70% of the data set as training data
set.seed(27)
train.indices <- sample(1:nrow(train), 0.7*nrow(train), replace=F)
train <- train[train.indices, ]
## select the other 30% as the testing data
test <- train[-train.indices,]


#------------------------ a predictive model ------------------------
#
# Build Random Forest Ensemble
set.seed(415)
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, data=train, importance=TRUE, ntree=50)
# Our prediction
prediction <- predict(fit, test)


# fit <- randomForest(Survived ~ ., data=train, importance=TRUE, ntree=500)
print(fit)


## MODEL EVALUATION
prediction <- predict(fit, test, type="response")
# calculate the confusion matrix
confusion <- table(prediction, test$Survived)
print(confusion)

print("-------------------------------")
# accuracy
accuracy <- sum(diag(confusion)) / sum(confusion)
paste("Acccuracy" ,accuracy, sep = " : ")

# precision
precision <- confusion[2,2] / sum(confusion[2,])
paste("Precision" ,precision, sep = " : ")

# recall
recall <- confusion[2,2] / sum(confusion[,2])
paste("   Recall" ,recall, sep = " : ")

# F1 score
F1 <- 2 * precision * recall / (precision + recall)
paste("       F1" ,F1, sep = " : ")
print("-------------------------------")

## We can also report probabilities
# prediction.prob <- predict(fit, test, type="prob")
# head(prediction.prob)
# head(test)

## show variable importance
importance(fit)
# varImpPlot(fit)


# # create a kaggle submission file
# submit <- data.frame(PassengerId = test$PassengerId, Survived = prediction)
# write.csv(submit, file = "firstforest.csv", row.names = FALSE)


# #------------------------ a predictive model ------------------------
# #
# # Build condition inference tree Random Forest
# set.seed(415)
# fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, data = train, controls=cforest_unbiased(ntree=2000, mtry=3))
# # Our prediction
# Prediction <- predict(fit, test, OOB=TRUE, type = "response")
# # create a kaggle submission file
# submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
# write.csv(submit, file = "ciforest.csv", row.names = FALSE)
