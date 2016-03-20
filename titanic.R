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
# train$Age[is.na(train$Age)] <- 24
# train$Fare2 <- '> 30'
# train$Fare2[train$Fare < 30 & train$Fare >= 20] <- '20-30'
# train$Fare2[train$Fare < 20 & train$Fare >= 10] <- '10-20'
# train$Fare2[train$Fare < 10] <- '< 10'
# train$Embarked <- sub("^$","?",train$Embarked)
# train$Name <- NULL
# train$Ticket <- NULL
# train$Cabin <- NULL
# train$Embarked <- as.factor(train$Embarked)
# train$Sex <- as.factor(train$Sex)
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

# Join together the test and train sets for easier feature engineering
test$Survived <- NA
combined <- rbind(train, test)

# Check what might be missing
#summary(combined)

# Convert to a string
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
famIDs <- famIDs[famIDs$Freq <= 2,]
combined$FamilyID[combined$FamilyID %in% famIDs$Var1] <- 'Small'
# Convert to a factor
combined$FamilyID <- factor(combined$FamilyID)


# New factor for Random Forests, only allowed <32 levels, so reduce number
combined$FamilyID2 <- combined$FamilyID
# Convert back to string
combined$FamilyID2 <- as.character(combined$FamilyID2)
combined$FamilyID2[combined$FamilySize <= 3] <- 'Small'
# And convert back to factor
combined$FamilyID2 <- factor(combined$FamilyID2)
#-------------------------------------------------------------------

# Fill in Age NAs
# summary(combined$Age)
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize, data=combined[!is.na(combined$Age),], method="anova")
combined$Age[is.na(combined$Age)] <- predict(Agefit, combined[is.na(combined$Age),])

# Fill in Embarked blanks
train$Embarked <- sub("^$","S",train$Embarked)
combined$Embarked <- factor(combined$Embarked)

# Fill in Fare NAs
#summary(combined$Fare)
#which(is.na(combined$Fare))
# some fares are 0.00, which is probably the same as NA
combined$Fare[combined$Fare == 0] <- NA

# A predictive model for missing fare way better than median value
#combined$Fare[is.na(combined$Fare)] <- median(combined$Fare, na.rm=TRUE)
Farefit <- rpart(Fare ~ Pclass + Sex + SibSp + Parch + Embarked + Title + FamilySize, data=combined[!is.na(combined$Fare),], method="anova")
combined$Fare[is.na(combined$Fare)] <- predict(Farefit, combined[is.na(combined$Fare),])


# remove unimportant features
combined <- combined[, -c(7,8,12,14,16)]


#  1...PassengerId
#  2...Survived
#  3...Pclass
#  4...Name
#  5...Sex
#  6...Age
#  7...SibSp
#  8...Parch
#  9...Ticket
# 10...Fare
# 11...Cabin
# 12...Embarked
# 13...Title
# 14...FamilySize
# 15...Surname
# 16...FamilyID
# 17...FamilyID2




# Split back into test and train sets
split <- nrow(train)
last <- nrow(combined) -1
train <- combined[1:split, ]
test <- combined[(split+1):last, ]

# or split the original training set
set.seed(27)
train.indices <- sample(1:nrow(train), 0.7*nrow(train), replace=F)
test <- train[-train.indices,]
train <- train[train.indices,]


#------------------------ a predictive model ------------------------
#
# Build Random Forest Ensemble
set.seed(415)
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2, data=train, importance=TRUE, ntree=2000)
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + Fare + Title + FamilyID2, data=train, importance=TRUE, ntree=2000)
# Look at variable importance
# varImpPlot(fit)

# Our prediction
prediction <- predict(fit, test)
# create a kaggle submission file
submit <- data.frame(PassengerId = test$PassengerId, Survived = prediction)
write.csv(submit, file = "firstforest.csv", row.names = FALSE)




fit <- randomForest(Survived ~ ., data=train, importance=TRUE, ntree=500)
prediction <- predict(fit, test, type="response")
titanic.rf.confusion <- table(prediction, test$Survived) # <----------------- ah, ha.
titanic.rf.accuracy <- sum(diag(titanic.rf.confusion)) / sum(titanic.rf.confusion)
titanic.rf.precision <- titanic.rf.confusion[2,2] / sum(titanic.rf.confusion[2,])

print(fit)
print(titanic.rf.confusion)
print(titanic.rf.accuracy)
print(titanic.rf.precision)
varImpPlot(fit)



#------------------------ a predictive model ------------------------
#
# Build condition inference tree Random Forest
set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, data = train, controls=cforest_unbiased(ntree=2000, mtry=3))
# Our prediction
Prediction <- predict(fit, test, OOB=TRUE, type = "response")
# create a kaggle submission file
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "ciforest.csv", row.names = FALSE)

print("done!")

