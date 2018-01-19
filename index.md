---
title: "Practical Machine Learning Course Project"
author: "Dorothy Buckley"
date: "January 14, 2018"
output: 
 html_document: 
    keep_md: yes
---



## Executive Summary
For this course project, I used the training and testing data sets extracted from the Weight Lifting Excercise Dataset.  The goal was to fit a model which would predict the manner in which 6 subjects did their exercises based on measurements taken from 4 accelerometers: belt, forearm, arm and dumbbell.  The *manner* is the "classe" variable from the training set: 1 of 5 possible classes.  I then used this model to predict 20 test cases.

Source information on this experiment can be found here: [link] (http://groupware.les.inf.puc-rio.br/har)

## Get Data

The training set has 160 variables and 19622 observations; the testing set has 160 variables and 20 observations.  Although they have the same number of variables, they are not exactly the same.  The training set has the variable called classe and the testing set has the variable called problem_id; each do not appear in the other sets.  Since we are trying to predict the classe variable for the testing set, I split the training data for testing prior to predicting on the original 20 test cases. 

## Clean Data

```r
del <- complete.cases(training)
###length(which(del[del==TRUE])) ## number of complete cases

####remove NAs
trngcount <- c(training[1,]!="NA")
x <- which(trngcount == TRUE)
newtrng <- training[,x]
colnumA <- ncol(newtrng)

### remove timestamp variables; visual inspection shows no correlation to timespan of repeated activity; remove x variable - index variable and no time correlation anyway
newtrngb <- newtrng[,c(2, 6:93)]
colnumB <- ncol(newtrngb) 

length(which(del[del==TRUE])) -> trngcc
```
Visual inspection of the csv file shows a large amount of NAs, and they are consistently dispersed through certain columns for the same records; in fact there are only 406 complete cases out of the 19622 records.  There are too many NAs for imputation so the NA's were removed which reduced to 93 variables.  *Note: Although I chose to use the Naive Bayes method which is supposed to handle NAs by omission or bypassing, the sheer number of NAs compelled me to use data cleaning prior to preprocessing.*There are 3 timestamp variables, however, there is no timespan of or cycle of repeated activity.  The 6 subjects seem to have performed these experiments one time.  So these three time variables, and an extraneous index variable "X", were also removed which reduced to 89 variables. 

## Split Dataset

```r
inTrain <- createDataPartition(y=newtrngb$classe, p=.75, list=FALSE)
trng <- newtrngb[inTrain,] ## 14718 records
tstg <- newtrngb[-inTrain,] ## 4904 records
nrow(trng) -> tnobs
nrow(tstg) -> ttobs
```
The training dataset was split into subsets to incorporate accuracy testing, prior to predicting on the actual test set which does not have the outcome variable "classe".  There are 14718 records for training and 4904 records for testing the model.

## Preprocess Training Data
### Remove Zero and Near-Zero Variance Predictors

```r
nzv <- nearZeroVar(trng)
nzvtrng <- trng[,-nzv] ## delete 0 and near-0 variance predictors from training table
dim(nzvtrng)[2] -> nzvvar
```
Next, the data was checked for predictors that only have a single unique value or only a handful of unique values occurring with very low frequencies to filter out of the modeling process. These zero-variance predictors may have an undue influence on the model and need to be discarded prior to modeling.  Preprocessing for these zero/near-zero variance predictors reduced the table size to 55 variables.  *Note: There are only 2 factor variables: user_name and classe.*

### Remove Highly Correlated and Linear Dependencies

```r
### table with no factors (took out user_name and classe)
nf <- nzvtrng[, 2:54]
### remove linear dependencies
trainingld <- findLinearCombos(nf)
a <- trainingld$linearCombos ## list of 0
b <- trainingld$remove ## NULL
###Highly correlated variables
trainingCor <- cor(nf)
highCor <- findCorrelation(trainingCor, cutoff = .8)
trngredux <- nzvtrng[,-highCor] ## delete highly correlated variables from training table
nfnames <- names(nf)
trngreduxnames <- names(trngredux)
##setdiff(nfnames, rednfnames)### highly correlated variables removed
cornum <- length(setdiff(nfnames, trngreduxnames))
setdiff(nfnames, trngreduxnames) -> cornumvar
ncol(trngredux) -> trngreduxvar
rm(newtrng, nf, nzvtrng, training, newtrngb) ##remove temporary tables from environment 
```
The factor variables were left out to run the functions for correlation and linear dependencies.  The 13  highly correlated variables removed are num_window, roll_belt, gyros_belt_z, accel_belt_x, accel_belt_y, total_accel_arm, gyros_arm_z, magnet_arm_x, total_accel_dumbbell, gyros_dumbbell_y, gyros_dumbbell_z, accel_dumbbell_y, gyros_forearm_x.  No linear dependencies were found.  The final number of variables after preprocessing is 42.

## Cross Validation

```r
set.seed(7272)
tc <- trainControl(method = "repeatedcv", number = 5, repeats = 5, preProcOptions = c("center", "scale"), allowParallel = TRUE)
```
The traincontrol option was used to specify the resampling method with repeated 5-fold cross-validation and for centering and scaling the data.

## Fit Model

```r
set.seed(3535)
ModFit1 <- train(classe ~ ., data = trngredux, method = "nb", trControl = tc)
stopCluster(cluster)
registerDoSEQ()
```
The train model chosen was Naive Bayes *(klaR package)* which is one of many models used for classification modeling. Although it may be less accurate than some other models, it is still considered a good baseline model.  The *naive* aspect is due to the underlying assumption of independence between predictors.  Highly correlated variables were already removed during pre-processing so this assumption was continued.

## Apply Equivalent Pre-processing to Test Data (Subsample of Training Data)

```r
tstgredux <- tstg[, trngreduxnames]
tstgreduxnames <- names(tstgredux)
##setdiff(trngreduxnames, tstgreduxnames) ran to ensure same variables in test and training sets
```
Running the functions for correlation and zero/near-zero variance may yield different variables on a test set.  Preprocessing the test set involved removing the same variables removed from the training set.

## Predict Test Data (Subsample of Training Data)

```r
set.seed(5252)
cluster2 <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster2)
tstgpred <- predict(ModFit1, tstgredux)
stopCluster(cluster2)
registerDoSEQ()
```

The following confusion matrix table compares the predicted values against the actual values of the test subsample of the training data set.

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1203  164  146  117   52
##          B   49  633   57    6   79
##          C   38   81  616  108   39
##          D   86   53   25  514   26
##          E   19   18   11   59  705
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.485726e-01   6.795487e-01   7.361827e-01   7.606659e-01   2.844617e-01 
## AccuracyPValue  McnemarPValue 
##   0.000000e+00   9.155801e-60
```
The accuracy came out to be 0.7485726. 

## Predict Test Data (Test Data: 20 Cases)

```r
## Apply Equivalent Pre-processing to Test Data
TESTreduxnames <- trngreduxnames[1:ncol(trngredux)-1]
TESTreduxnamesInd <- names(testing) %in% TESTreduxnames
TESTredux <- testing[, TESTreduxnamesInd]
TESTreduxnames <- names(TESTredux)
##setdiff(TESTreduxnames, names(testing)) check to see variables match
## Run Test Data through Model
set.seed(3838)
TESTpred <- predict(ModFit1, TESTredux)
```
 
## Test Data Prediction Results (Test Data: 20 Cases)
The following classes are predicted for the 20 test cases: A, A, A, A, A, C, D, E, A, A, A, A, B, A, E, E, A, B, A, B.  

Since there is no "classe" variable attached to the actual test data, there is no way to compute the accuracy of these test case predictions.  It can only be assumed to be lower than the accuracy for the testing portion of the training data, but it will hopefully be close.
