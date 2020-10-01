#Loading required packages
if (!require("pacman")) install.packages("pacman","caret","e1071","lift")
pacman::p_load(pacman,rio,tidyverse,caret,e1071,lift)

df <- import("indianliver.xlsx")
nrow(df)
str(df)
df$Result <- factor(ifelse(df$Dataset == 2,0,1 ))
df$Gender.d <- factor(ifelse(df$Gender == "Male",1,0 ))
df <- df[,-c(2,11)]
str(df)

# handling missing data
sum(is.na(df))
library(mice)

md.pattern(df)
imputed.df <- mice(data = df, m=5, method = "pmm", maxit = 50, seed = 500)
summary(imputed.df)
imputed.df$imp$Albumin_and_Globulin_Ratio
completed.df <- complete(imputed.df,2)

# handling outliers
mod <- glm(Result ~ ., data = completed.df, family = "binomial")
cooksd <- cooks.distance(mod)
influential <- as.numeric(names(cooksd)[cooksd > 4*mean(cooksd, na.rm = 'T')])
influential
completed.df <- completed.df[-influential,]
nrow(completed.df)

# Partitioning

training.rows <- sample(rownames(completed.df),dim(completed.df)[1]*0.5)
valid.rows <- sample(setdiff(rownames(completed.df),training.rows),dim(completed.df)[1]*0.3)
test.rows <- setdiff(rownames(completed.df), union(training.rows,valid.rows))

train.data <- completed.df[training.rows,]
nrow(train.data)
valid.data <- completed.df[valid.rows,]
nrow(valid.data)
test.data <- completed.df[test.rows,]
nrow(test.data)

# Applying data mining methods

#0. Baseline classification
#Every candidate is considered as liver patient.

temp <- completed.df[completed.df$Result==0,]
nrow(temp)
nrow(completed.df)
# Accuracy 
nrow(temp)/nrow(completed.df)
baseline.accuracy <- 1-(nrow(temp)/nrow(completed.df))
baseline.accuracy

#1. Logistic Regression

logit.model <- glm(Result ~ Age + Gender.d + Total_Bilirubin + Direct_Bilirubin 
                   + Alkaline_Phosphotase + Alamine_Aminotransferase 
                   + Alamine_Aminotransferase+ Total_Protiens + Albumin
                   + Albumin_and_Globulin_Ratio, data = train.data,
                   family = "binomial")

summary(logit.model)

# Applying model on Validation Set
validation.prob <- predict(logit.model, valid.data, type = "response")
validation.classifications <- ifelse(validation.prob < 0.5, 0, 1)
confusionMatrix(as.factor(validation.classifications),
                as.factor(valid.data$Result),
                positive = "1")

# Applying model on test Set
test.prob <- predict(logit.model, test.data, type = "response")
test.classifications <- ifelse(test.prob < 0.5, 0, 1)
confusionMatrix(as.factor(test.classifications),
                as.factor(test.data$Result),
                positive = "1")

# 2. Classification Trees

library(rpart)

liver.full.tree <- rpart(Result ~ Age + Gender.d + Total_Bilirubin + Direct_Bilirubin 
                         + Alkaline_Phosphotase + Alamine_Aminotransferase 
                         + Alamine_Aminotransferase+ Total_Protiens + Albumin
                         + Albumin_and_Globulin_Ratio, data = train.data, method = "class",
                         control = rpart.control(cp = 0.0, minsplit = 0))

library(rattle)
fancyRpartPlot(liver.full.tree)

printcp(liver.full.tree)

prunned.liver.tree <- prune(liver.full.tree, cp = 0.015)
fancyRpartPlot(prunned.liver.tree)

# Applying model on training data
actual.liver.train <- train.data$Result
predict.liver.train <- predict(prunned.liver.tree, type = "class")
table(actual.liver.train,predict.liver.train)
#accuracy
sum(actual.liver.train == predict.liver.train)/nrow(train.data)

# Applying model on validation data
actual.liver.valid <- valid.data$Result
predict.liver.valid <- predict(prunned.liver.tree, type = "class", newdata = valid.data)
table(actual.liver.valid,predict.liver.valid)
#accuracy
sum(actual.liver.valid == predict.liver.valid)/nrow(valid.data)

# Applying model on test data
actual.liver.test <- test.data$Result
predict.liver.test <- predict(prunned.liver.tree, type = "class", newdata = test.data)
table(actual.liver.test,predict.liver.test)
#accuracy
sum(actual.liver.test == predict.liver.test)/nrow(test.data)

#3. Random Forests
# ntree = 25 gives the optimal accuracy for validation as well as test data.

library(randomForest)

# ntree = 5
liver.forest.model.5 <- randomForest(Result ~ Age + Gender.d + Total_Bilirubin + Direct_Bilirubin 
                                     + Alkaline_Phosphotase + Alamine_Aminotransferase 
                                     + Alamine_Aminotransferase+ Total_Protiens + Albumin
                                     + Albumin_and_Globulin_Ratio, data=train.data, ntree=5)

# Applying model on training data
predict.liver.train.rf5 <- predict(liver.forest.model.5, newdata = train.data)
sum(actual.liver.train == predict.liver.train.rf5)/nrow(train.data)

# Applying model on validation data
predict.liver.valid.rf5 <- predict(liver.forest.model.5, newdata = valid.data)
sum(actual.liver.valid == predict.liver.valid.rf5)/nrow(valid.data)

# Applying model on test data
predict.liver.test.rf5 <- predict(liver.forest.model.5, newdata = test.data)
sum(actual.liver.test == predict.liver.test.rf5)/nrow(test.data)

# ntree = 11
liver.forest.model.11 <- randomForest(Result ~ Age + Gender.d + Total_Bilirubin + Direct_Bilirubin 
                                      + Alkaline_Phosphotase + Alamine_Aminotransferase 
                                      + Alamine_Aminotransferase+ Total_Protiens + Albumin
                                      + Albumin_and_Globulin_Ratio, data=train.data, ntree=11)

# Applying model on training data
predict.liver.train.rf11 <- predict(liver.forest.model.11, newdata = train.data)
sum(actual.liver.train == predict.liver.train.rf11)/nrow(train.data)

# Applying model on validation data
predict.liver.valid.rf11 <- predict(liver.forest.model.11, newdata = valid.data)
sum(actual.liver.valid == predict.liver.valid.rf11)/nrow(valid.data)

# Applying model on test data
predict.liver.test.rf11 <- predict(liver.forest.model.11, newdata = test.data)
sum(actual.liver.test == predict.liver.test.rf11)/nrow(test.data)

# ntree = 25
liver.forest.model.25 <- randomForest(Result ~ Age + Gender.d + Total_Bilirubin + Direct_Bilirubin 
                                      + Alkaline_Phosphotase + Alamine_Aminotransferase 
                                      + Alamine_Aminotransferase+ Total_Protiens + Albumin
                                      + Albumin_and_Globulin_Ratio, data=train.data, ntree=25)

# Applying model on training data
predict.liver.train.rf25 <- predict(liver.forest.model.25, newdata = train.data)
sum(actual.liver.train == predict.liver.train.rf25)/nrow(train.data)

# Applying model on validation data
predict.liver.valid.rf25 <- predict(liver.forest.model.25, newdata = valid.data)
sum(actual.liver.valid == predict.liver.valid.rf25)/nrow(valid.data)

# Applying model on test data
predict.liver.test.rf25 <- predict(liver.forest.model.25, newdata = test.data)
sum(actual.liver.test == predict.liver.test.rf25)/nrow(test.data)

# ntree = 101
liver.forest.model.101 <- randomForest(Result ~ Age + Gender.d + Total_Bilirubin + Direct_Bilirubin 
                                       + Alkaline_Phosphotase + Alamine_Aminotransferase 
                                       + Alamine_Aminotransferase+ Total_Protiens + Albumin
                                       + Albumin_and_Globulin_Ratio, data=train.data, ntree=101)

# Applying model on training data
predict.liver.train.rf101 <- predict(liver.forest.model.101, newdata = train.data)
sum(actual.liver.train == predict.liver.train.rf101)/nrow(train.data)

# Applying model on validation data
predict.liver.valid.rf101 <- predict(liver.forest.model.101, newdata = valid.data)
sum(actual.liver.valid == predict.liver.valid.rf101)/nrow(valid.data)

# Applying model on test data
predict.liver.test.rf101 <- predict(liver.forest.model.101, newdata = test.data)
sum(actual.liver.test == predict.liver.test.rf101)/nrow(test.data)

# ntree = 401
liver.forest.model.401 <- randomForest(Result ~ Age + Gender.d + Total_Bilirubin + Direct_Bilirubin 
                                       + Alkaline_Phosphotase + Alamine_Aminotransferase 
                                       + Alamine_Aminotransferase+ Total_Protiens + Albumin
                                       + Albumin_and_Globulin_Ratio, data=train.data, ntree=401)

# Applying model on training data
predict.liver.train.rf401 <- predict(liver.forest.model.401, newdata = train.data)
sum(actual.liver.train == predict.liver.train.rf401)/nrow(train.data)

# Applying model on validation data
predict.liver.valid.rf401 <- predict(liver.forest.model.401, newdata = valid.data)
sum(actual.liver.valid == predict.liver.valid.rf401)/nrow(valid.data)

# Applying model on test data
predict.liver.test.rf401 <- predict(liver.forest.model.401, newdata = test.data)
sum(actual.liver.test == predict.liver.test.rf401)/nrow(test.data)

# ntree = 1601
liver.forest.model.1601 <- randomForest(Result ~ Age + Gender.d + Total_Bilirubin + Direct_Bilirubin 
                                        + Alkaline_Phosphotase + Alamine_Aminotransferase 
                                        + Alamine_Aminotransferase+ Total_Protiens + Albumin
                                        + Albumin_and_Globulin_Ratio, data=train.data, ntree=1601)

# Applying model on training data
predict.liver.train.rf1601 <- predict(liver.forest.model.1601, newdata = train.data)
sum(actual.liver.train == predict.liver.train.rf1601)/nrow(train.data)

# Applying model on validation data
predict.liver.valid.rf1601 <- predict(liver.forest.model.1601, newdata = valid.data)
sum(actual.liver.valid == predict.liver.valid.rf1601)/nrow(valid.data)

# Applying model on test data
predict.liver.test.rf1601 <- predict(liver.forest.model.1601, newdata = test.data)
sum(actual.liver.test == predict.liver.test.rf1601)/nrow(test.data)

#4. Boosting
# iter = 25 gives the optimal accuracy for validation and testing data.

library(ada)

# iter =5
liver.boost.model.5 <- ada(Result ~ Age + Gender.d + Total_Bilirubin + Direct_Bilirubin 
                           + Alkaline_Phosphotase + Alamine_Aminotransferase 
                           + Alamine_Aminotransferase+ Total_Protiens + Albumin
                           + Albumin_and_Globulin_Ratio , data = train.data, iter=5)

#Applying model on training data
predict.liver.train.boost5 <- predict(liver.boost.model.5, newdata = train.data)
sum(actual.liver.train == predict.liver.train.boost5)/nrow(train.data)

#Applying model on validation data
predict.liver.valid.boost5 <- predict(liver.boost.model.5, newdata = valid.data)
sum(actual.liver.valid == predict.liver.valid.boost5)/nrow(valid.data)

#Applying model on test data
predict.liver.test.boost5 <- predict(liver.boost.model.5, newdata = test.data)
sum(actual.liver.test == predict.liver.test.boost5)/nrow(test.data)

# iter =11
liver.boost.model.11 <- ada(Result ~ Age + Gender.d + Total_Bilirubin + Direct_Bilirubin 
                            + Alkaline_Phosphotase + Alamine_Aminotransferase 
                            + Alamine_Aminotransferase+ Total_Protiens + Albumin
                            + Albumin_and_Globulin_Ratio , data = train.data, iter=11)

#Applying model on training data
predict.liver.train.boost11 <- predict(liver.boost.model.11, newdata = train.data)
sum(actual.liver.train == predict.liver.train.boost11)/nrow(train.data)

#Applying model on validation data
predict.liver.valid.boost11 <- predict(liver.boost.model.11, newdata = valid.data)
sum(actual.liver.valid == predict.liver.valid.boost11)/nrow(valid.data)

#Applying model on test data
predict.liver.test.boost11 <- predict(liver.boost.model.11, newdata = test.data)
sum(actual.liver.test == predict.liver.test.boost11)/nrow(test.data)

# iter =25
liver.boost.model.25 <- ada(Result ~ Age + Gender.d + Total_Bilirubin + Direct_Bilirubin 
                            + Alkaline_Phosphotase + Alamine_Aminotransferase 
                            + Alamine_Aminotransferase+ Total_Protiens + Albumin
                            + Albumin_and_Globulin_Ratio , data = train.data, iter=25)

#Applying model on training data
predict.liver.train.boost25 <- predict(liver.boost.model.25, newdata = train.data)
sum(actual.liver.train == predict.liver.train.boost25)/nrow(train.data)

#Applying model on validation data
predict.liver.valid.boost25 <- predict(liver.boost.model.25, newdata = valid.data)
sum(actual.liver.valid == predict.liver.valid.boost25)/nrow(valid.data)

#Applying model on test data
predict.liver.test.boost25 <- predict(liver.boost.model.25, newdata = test.data)
sum(actual.liver.test == predict.liver.test.boost25)/nrow(test.data)

# iter =101
liver.boost.model.101 <- ada(Result ~ Age + Gender.d + Total_Bilirubin + Direct_Bilirubin 
                             + Alkaline_Phosphotase + Alamine_Aminotransferase 
                             + Alamine_Aminotransferase+ Total_Protiens + Albumin
                             + Albumin_and_Globulin_Ratio , data = train.data, iter=101)

#Applying model on training data
predict.liver.train.boost101 <- predict(liver.boost.model.101, newdata = train.data)
sum(actual.liver.train == predict.liver.train.boost101)/nrow(train.data)

#Applying model on validation data
predict.liver.valid.boost101 <- predict(liver.boost.model.101, newdata = valid.data)
sum(actual.liver.valid == predict.liver.valid.boost101)/nrow(valid.data)

#Applying model on test data
predict.liver.test.boost101 <- predict(liver.boost.model.101, newdata = test.data)
sum(actual.liver.test == predict.liver.test.boost101)/nrow(test.data)

# iter =401
liver.boost.model.401 <- ada(Result ~ Age + Gender.d + Total_Bilirubin + Direct_Bilirubin 
                             + Alkaline_Phosphotase + Alamine_Aminotransferase 
                             + Alamine_Aminotransferase+ Total_Protiens + Albumin
                             + Albumin_and_Globulin_Ratio , data = train.data, iter=401)

#Applying model on training data
predict.liver.train.boost401 <- predict(liver.boost.model.401, newdata = train.data)
sum(actual.liver.train == predict.liver.train.boost401)/nrow(train.data)

#Applying model on validation data
predict.liver.valid.boost401 <- predict(liver.boost.model.401, newdata = valid.data)
sum(actual.liver.valid == predict.liver.valid.boost401)/nrow(valid.data)

#Applying model on test data
predict.liver.test.boost401 <- predict(liver.boost.model.401, newdata = test.data)
sum(actual.liver.test == predict.liver.test.boost401)/nrow(test.data)

# iter =1601
liver.boost.model.1601 <- ada(Result ~ Age + Gender.d + Total_Bilirubin + Direct_Bilirubin 
                              + Alkaline_Phosphotase + Alamine_Aminotransferase 
                              + Alamine_Aminotransferase+ Total_Protiens + Albumin
                              + Albumin_and_Globulin_Ratio , data = train.data, iter=1601)

#Applying model on training data
predict.liver.train.boost1601 <- predict(liver.boost.model.1601, newdata = train.data)
sum(actual.liver.train == predict.liver.train.boost1601)/nrow(train.data)

#Applying model on validation data
predict.liver.valid.boost1601 <- predict(liver.boost.model.1601, newdata = valid.data)
sum(actual.liver.valid == predict.liver.valid.boost1601)/nrow(valid.data)

#Applying model on test data
predict.liver.test.boost1601 <- predict(liver.boost.model.1601, newdata = test.data)
sum(actual.liver.test == predict.liver.test.boost1601)/nrow(test.data)
