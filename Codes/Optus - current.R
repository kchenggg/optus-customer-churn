#response variable: churn_ind 
setwd("C:/Users/Kerri C/Desktop")

#Import the data
#set stringsAsFactors=FALSE to  replace NAs with "NONE" first & recode data type
myData <- read.csv('NewDataset v2.csv', header = TRUE, stringsAsFactors = FALSE)
attach(myData)
head(myData)
str(myData) #check data type in dataset
colSums(is.na(myData)) #check missing values in dataset for columns

myDataNew <- myData
attach(myDataNew)

#Change NA to "NONE" only for cols with missing values (categorical)
myDataNew$ACCT_SERVICES_SEGMENT_LVL1 [which(is.na(ACCT_SERVICES_SEGMENT_LVL1))] <- "NONE"
myDataNew$ACCT_SERVICES_SEGMENT_LVL2 [which(is.na(ACCT_SERVICES_SEGMENT_LVL2))] <- "NONE"
myDataNew$PREV_PLAN_START_MOC [which(is.na(PREV_PLAN_START_MOC))] <- "NONE"
myDataNew$LANDSCAPE_GROUP [which(is.na(LANDSCAPE_GROUP))] <- "NONE"
myDataNew$LANDSCAPE_SEGMENT [which(is.na(LANDSCAPE_SEGMENT))] <- "NONE"
myDataNew$LIFESTAGE_SEGMENT_CD [which(is.na(LIFESTAGE_SEGMENT_CD))] <- "NONE"
myDataNew$BILL_STMT_VOICE_TIER_USED [which(is.na(BILL_STMT_VOICE_TIER_USED))] <- "NONE"
myDataNew$BILL_STMT_DATA_TIER_USED [which(is.na(BILL_STMT_DATA_TIER_USED))] <- "NONE"
myDataNew$BILL_STMT_PRINCPL_SVC_IND [which(is.na(BILL_STMT_PRINCPL_SVC_IND))] <- "NONE"
myDataNew$COMMERCIAL_RECONTRACT_DT [which(is.na(COMMERCIAL_RECONTRACT_DT))] <- "NONE"
myDataNew$COMMERCIAL_CONTRACT_DURATION [which(is.na(COMMERCIAL_CONTRACT_DURATION))] <- "NONE"
myDataNew$CHURNED_NEXT_MTH [which(is.na(CHURNED_NEXT_MTH))] <- "NONE"
myDataNew$RECON_SMS_NEXT_MTH [which(is.na(RECON_SMS_NEXT_MTH))] <- "NONE"
myDataNew$RECON_TELE_NEXT_MTH [which(is.na(RECON_TELE_NEXT_MTH))] <- "NONE"
myDataNew$RECON.EMAIL [which(is.na(RECON.EMAIL))] <- "NONE"
colSums(is.na(myDataNew)) 


  #myDataNew$REPORTING_MONTH <- as.factor(myDataNew$REPORTING_MONTH)
  #myDataNew$REPORTING_MONTH_END <- as.factor(myDataNew$REPORTING_MONTH_END)
  #myDataNew$ACCT_PROD_CNT_SERVICES <- as.numeric(myDataNew$ACCT_PROD_CNT_SERVICES)
  #myDataNew$BILL_ZIP <- as.numeric(myDataNew$BILL_ZIP)
  #myDataNew$SERVICE_TENURE <- as.numeric(myDataNew$SERVICE_TENURE)
  #myDataNew$PLAN_ACCESS_FEE <- as.numeric(myDataNew$PLAN_ACCESS_FEE)
  #myDataNew$PLAN_ACCESS_FEE_INCL_GST <- as.numeric(myDataNew$PLAN_ACCESS_FEE_INCL_GST)
  #myDataNew$PLAN_TENURE <- as.numeric(myDataNew$PLAN_TENURE)
  #myDataNew$MONTH_OF_CONTRACT <- as.numeric(myDataNew$MONTH_OF_CONTRACT)
  #myDataNew$MONTH_OF_CONTRACT_COMPLETED <- as.numeric(myDataNew$MONTH_OF_CONTRACT_COMPLETED)
  #myDataNew$MONTHS_OF_CONTRACT_REMAINING <- as.numeric(myDataNew$MONTHS_OF_CONTRACT_REMAINING)
  #myDataNew$MTHS_SINCE_LAST_FX_CON_START <- as.numeric(myDataNew$MTHS_SINCE_LAST_FX_CON_START)
  #myDataNew$MTHS_SINCE_LAST_FX_CON_END <- as.numeric(myDataNew$MTHS_SINCE_LAST_FX_CON_END)
  #myDataNew$LAST_FX_CONTRACT_DURATION <- as.numeric(myDataNew$LAST_FX_CONTRACT_DURATION)
  #myDataNew$PREV_CONTRACT_DURATION <- as.numeric(myDataNew$PREV_CONTRACT_DURATION)
  #myDataNew$HANDSET_USED_BRAND <- as.factor(myDataNew$HANDSET_USED_BRAND)
  #myDataNew$HANDSET_USED_MODEL_LVL1 <- as.factor(myDataNew$HANDSET_USED_MODEL_LVL1)
  #myDataNew$HANDSET_SOLD_MODEL_LVL1 <- as.factor(myDataNew$HANDSET_SOLD_MODEL_LVL1)
  #myDataNew$CHANNEL_GROUPING_LVL1 <- as.factor(myDataNew$CHANNEL_GROUPING_LVL1)
  #myDataNew$CHANNEL_GROUPING_LVL2 <- as.factor(myDataNew$CHANNEL_GROUPING_LVL2)
  #myDataNew$MONTHLY_SPEND <- as.numeric(myDataNew$MONTHLY_SPEND)
  #myDataNew$MONTHLY_SPEND_GST <- as.numeric(myDataNew$MONTHLY_SPEND_GST)
  #myDataNew$MONTHLY_RATED_VOICE_CALLS <- as.numeric(myDataNew$MONTHLY_RATED_VOICE_CALLS)
  #myDataNew$MONTHLY_RATED_VOICE_DURATION <- as.numeric(myDataNew$MONTHLY_RATED_VOICE_DURATION)
  #myDataNew$MONTHLY_RATED_VOICE_CHARGE <- as.numeric(myDataNew$MONTHLY_RATED_VOICE_CHARGE)
  #myDataNew$MONTHLY_RATED_SMS_CALLS <- as.numeric(myDataNew$MONTHLY_RATED_SMS_CALLS)
  #myDataNew$MONTHLY_RATED_SMS_CHARGE <- as.numeric(myDataNew$MONTHLY_RATED_SMS_CHARGE)
  #myDataNew$MONTHLY_RATED_DATA_VOLUME <- as.numeric(myDataNew$MONTHLY_RATED_DATA_VOLUME)
  #myDataNew$MONTHLY_RATED_DATA_CHARGE <- as.numeric(myDataNew$MONTHLY_RATED_DATA_CHARGE)
  #myDataNew$CHURN_IND.1 <- as.numeric(myDataNew$CHURN_IND.1)

  

#Encoding data to correct data types for replaced values
myDataNew$ACCT_SERVICES_SEGMENT_LVL1 <- as.factor(myDataNew$ACCT_SERVICES_SEGMENT_LVL1) 
myDataNew$ACCT_SERVICES_SEGMENT_LVL2 <- as.factor(myDataNew$ACCT_SERVICES_SEGMENT_LVL2)
myDataNew$PREV_PLAN_START_MOC <- as.numeric(myDataNew$PREV_PLAN_START_MOC)
myDataNew$LANDSCAPE_GROUP <- as.factor(myDataNew$LANDSCAPE_GROUP)
myDataNew$LANDSCAPE_SEGMENT <- as.factor(myDataNew$LANDSCAPE_SEGMENT)
myDataNew$LIFESTAGE_SEGMENT_CD <- as.factor(myDataNew$LIFESTAGE_SEGMENT_CD)
myDataNew$BILL_STMT_VOICE_TIER_USED <- as.factor(myDataNew$BILL_STMT_VOICE_TIER_USED)
myDataNew$BILL_STMT_DATA_TIER_USED <- as.integer(myDataNew$BILL_STMT_DATA_TIER_USED)
myDataNew$BILL_STMT_PRINCPL_SVC_IND <- as.factor(myDataNew$BILL_STMT_PRINCPL_SVC_IND)
myDataNew$COMMERCIAL_RECONTRACT_DT <- as.factor(myDataNew$COMMERCIAL_RECONTRACT_DT)
myDataNew$COMMERCIAL_CONTRACT_DURATION <- as.factor(myDataNew$COMMERCIAL_CONTRACT_DURATION)
myDataNew$CHURNED_NEXT_MTH <- as.factor(myDataNew$CHURNED_NEXT_MTH)
myDataNew$RECON_SMS_NEXT_MTH <- as.factor(myDataNew$RECON_SMS_NEXT_MTH)
myDataNew$RECON_TELE_NEXT_MTH <- as.factor(myDataNew$RECON_TELE_NEXT_MTH)
myDataNew$RECON.EMAIL <- as.factor(myDataNew$RECON.EMAIL)
#Encoding data to correct data types for char due to stringsAsFactors=FALSE 
myDataNew$HANDSET_USED_BRAND <- as.factor(myDataNew$HANDSET_USED_BRAND)
myDataNew$HANDSET_USED_MODEL_LVL1 <- as.factor(myDataNew$HANDSET_USED_MODEL_LVL1)
myDataNew$HANDSET_SOLD_MODEL_LVL1 <- as.factor(myDataNew$HANDSET_SOLD_MODEL_LVL1)
myDataNew$CHANNEL_GROUPING_LVL1 <- as.factor(myDataNew$CHANNEL_GROUPING_LVL1)
myDataNew$CHANNEL_GROUPING_LVL2 <- as.factor(myDataNew$CHANNEL_GROUPING_LVL2)
myDataNew$PRODUCT <- as.factor(myDataNew$PRODUCT)
#Encoding data to correct data types
myDataNew$CUST_ID <- as.numeric(myDataNew$CUST_ID)
myDataNew$ACCOUNT_TENURE <- as.numeric(myDataNew$ACCOUNT_TENURE)
myDataNew$ACCT_CNT_SERVICES<-as.numeric(myDataNew$ACCT_CNT_SERVICES)
myDataNew$ACCOUNT_CATEGORY <- as.factor(myDataNew$ACCOUNT_CATEGORY)
myDataNew$AGE <- as.numeric(myDataNew$AGE)
myDataNew$CFU <- as.factor(myDataNew$CFU)
myDataNew$ADD_IND <- as.factor(myDataNew$ADD_IND)
myDataNew$CHURN_IND <- as.factor(myDataNew$CHURN_IND)
myDataNew$RECONTRACT_IND <- as.factor(myDataNew$RECONTRACT_IND)
myDataNew$PLAN_CHANGE_IND <- as.factor(myDataNew$PLAN_CHANGE_IND)
myDataNew$CONTRACT_CHANGE_IND <- as.factor(myDataNew$CONTRACT_CHANGE_IND)
myDataNew$COMMERCIAL_RECONTRACT_IND <- as.factor(myDataNew$COMMERCIAL_RECONTRACT_IND)
myDataNew$CURRENT_CONTRACT_DURATION <- as.factor(myDataNew$CURRENT_CONTRACT_DURATION)
myDataNew$COUNTRY_METRO_REGION <- as.factor(myDataNew$COUNTRY_METRO_REGION)
myDataNew$STATE <- as.factor(myDataNew$STATE)
myDataNew$CONTRACT_STATUS <- as.factor(myDataNew$CONTRACT_STATUS)
myDataNew$BYO_PLAN_STATUS <- as.factor(myDataNew$BYO_PLAN_STATUS)
myDataNew$HANDSET_USED_CAPABILITY <- as.factor(myDataNew$HANDSET_USED_CAPABILITY)
myDataNew$HANDSET_SOLD_CAPABILITY <- as.factor(myDataNew$HANDSET_SOLD_CAPABILITY)
myDataNew$HANDSET_SOLD_BRAND <- as.factor(myDataNew$HANDSET_SOLD_BRAND)

#Populate empty cells with column mean - numeric columns
for(i in 1:ncol(myDataNew))
{
  myDataNew[is.na(myDataNew[,i]), i] <- mean(myDataNew[,i], na.rm = TRUE)
}

#remove col 
myDataNew$CHURN_IND.1 <- NULL #remove duplicate col
myDataNew$REPORTING_MONTH <- NULL 
myDataNew$REPORTING_MONTH_END <- NULL 
#myDataNew$COMMERCIAL_RECONTRACT_DT <- NULL 
myDataNew$CUST_ID <- NULL #cus_id

#remove rows
myDataNew <- myDataNew[!is.na(myDataNew$STATE), ]
myDataNew <- myDataNew[!grepl("FF", myDataNew$LIFESTAGE_SEGMENT_CD), ]
myDataNew <- myDataNew[!grepl("MISSING CONTRACT", myDataNew$CONTRACT_STATUS), ]
#rownames(myDataNew) <- NULL

str(myDataNew)
colSums(is.na(myDataNew))

#split the data into 70% training, 30% testing. We partition the dataset for random sampling using caret
#install.packages("pbkrtest")
#install.packages("caret")
library(caret) 

'%ni%' <- Negate('%in%')  # define 'not in' func
options(scipen=999)  # prevents printing scientific notations.
set.seed(100)

inTrain <- createDataPartition(myDataNew$CHURN_IND, p = 0.7, list = FALSE)
trainData <- myDataNew[inTrain, ]
#write.csv(myDataNew, file = "TRaindata.csv")
testData <- myDataNew[-inTrain, ]
#write.csv(myDataNew, file = "TEstdata AAA.csv")
#Export to excel
#write.csv(myDataNew, file = "NewDataset - cleansed current.csv")





#myDataNew$BILL_STMT_VOICE_TIER_USED <- NULL 
#myDataNew$BILL_STMT_DATA_TIER_USED <- NULL 
#myDataNew$BILL_STMT_PRINCPL_SVC_IND <- NULL 
#myDataNew$COMMERCIAL_CONTRACT_DURATION <- NULL 
#myDataNew$COMMERCIAL_RECONTRACT_DT <- NULL 
#myDataNew$LANDSCAPE_GROUP <- NULL 
#myDataNew$LANDSCAPE_SEGMENT <- NULL 
#myDataNew$LIFESTAGE_SEGMENT_CD <- NULL 
#myDataNew$PREV_PLAN_START_MOC <- NULL 
#myDataNew$ACCT_SERVICES_SEGMENT_LVL1 <- NULL 
#myDataNew$ACCT_SERVICES_SEGMENT_LVL2 <- NULL 

#remove rows
#myDataNew <- myDataNew[!(myDataNew$LIFESTAGE_SEGMENT_CD=="FF"), ]
#d=with(myDataNew, !is.na(STATE) & !is.na(COUNTRY_METRO_REGION) & !is.na(MONTH_OF_CONTRACT) & !is.na(MONTH_OF_CONTRACT_COMPLETED))
#myDataNew = myDataNew[d,]
#colSums(is.na(myDataNew)) #check missing values in dataset for columns






###1. Regression 
#feature selection - variable importance###
fit_reg <- glm(CHURN_IND ~., family = binomial(link='logit'), maxit = 100, myDataNew)
summary(fit_reg) #shows sig. variables
anova(fit_reg, test="Chisq")
#predict on test data
pred_LR <- predict(fit_reg, testData, type = "response")
pred_LR
#accuracy rate
#confusionMatrix(pred_LR, testData$CHURN_IND)
confusionMatrix(table(pred_LR, testData$CHURN_IND))


fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != test$Survived)
print(paste('Accuracy',1-misClasificError))















###1. RandomForest### (each tree is built on 70% of training. As the forest is built, each
  #tree is trained on samples not used to built the tree, hence OOB error rate is the
  #internal error estimate (misclassification rate) as it is being constructed)
#can be modelled for categorical variables
#won't overfit the model if there are enough trees
#can handle missing values
install.packages("randomForest")
library(randomForest)
set.seed(100)
fit_random <- randomForest(CHURN_IND ~ ., trainData, importance=TRUE, confusion=TRUE)
  #error if you don't use as.factor. It predicts a regression since some variables are of wrong data type
#plot(fit_random) to see best no. of trees and their error rate
plot(fit_random)

#predict on test data
pred_random <- predict(fit_random, testData)

#confusion matrix
#confusemRF <- table(pred_random, testData$CHURN_IND)
#colnames(confusemRF) <- c('Predict Bad','Predict Good')
#rownames(confusemRF) <- c('Actual Bad','Actual Good')
#confusemRF

#error rate
  #err <- 100 * (1 - (sum(diag(confusemRF))/sum(confusemRF)))
  #err
confusionMatrix(pred_random, testData$CHURN_IND)

#variable importance 
varImpPlot(fit_random) 
importance(fit_random) 
varImpPlot(fit_random, sort = T, n.var = 10, main="Top 10 variables")

#selecting variables
fit_random2 <- randomForest(CHURN_IND ~ 
                          RECON_TELE_NEXT_MTH + 
                          RECON.EMAIL +
                          RECON_SMS_NEXT_MTH + 
                          CHURNED_NEXT_MTH + 
                          MONTHLY_SPEND_GST + 
                          MONTHLY_SPEND + 
                          BILL_STMT_DATA_TIER_USED + 
                          MONTHLY_RATED_VOICE_CALLS + 
                          MONTHLY_RATED_VOICE_DURATION + 
                          MONTHS_OF_CONTRACT_REMAINING + 
                          BILL_STMT_PRINCPL_SVC_IND + 
                          MONTHLY_RATED_DATA_VOLUME + 
                          CONTRACT_STATUS + 
                          MONTHLY_RATED_SMS_CALLS + 
                          MTHS_SINCE_LAST_FX_CON_START + 
                          CURRENT_CONTRACT_DURATION + 
                          PLAN_ACCESS_FEE + 
                          PLAN_ACCESS_FEE_INCL_GST + 
                          PLAN_TENURE + 
                          BILL_STMT_VOICE_TIER_USED + 
                          HANDSET_SOLD_MODEL_LVL1 + 
                          HANDSET_SOLD_BRAND + 
                          MONTH_OF_CONTRACT + 
                          HANDSET_SOLD_CAPABILITY + 
                          SERVICE_TENURE + 
                          MONTH_OF_CONTRACT_COMPLETED + 
                          HANDSET_USED_MODEL_LVL1 + 
                          ACCT_CNT_SERVICES + 
                          LAST_FX_CONTRACT_DURATION + 
                          CONTRACT_CHANGE_IND,
                          trainData, importance=TRUE, confusion=TRUE)
plot(fit_random2)
pred_random2 <- predict(fit_random2, testData)
confusionMatrix(pred_random2, testData$CHURN_IND)


#No. of nodes for the trees
hist(treesize(fit_random), main="No. of nodes for the trees")

#fine tuning Random Forest with the best mtry, the no. of variables for splitting at each tree node. Exclude response variable.
  #change in 1.5 ratio, include those that changess with 0.1 improvement 
newRF <- tuneRF (trainData[,-1], trainData[,1], ntreeTry = 500, stepFactor = 1.5, 
                    improve = 0.1, trace = TRUE, plot = TRUE)
print(newRF)





###2. SVM### uses hyperplane which classifies each into their own category
library(e1071)
set.seed(100)
fitSVM <- svm(CHURN_IND ~ ., trainData)
summary(fitSVM)

#predict on test data 
predictSVM <- predict(fitSVM, testData)

#confuseSVM <- table(predictSVM, testData$CHURN_IND)
#colnames(confuseSVM) <- c('Predict Bad','Predict Good')
#rownames(confuseSVM) <- c('Actual Bad','Actual Good')
#confuseSVM

#confusion matrix. Error rate 
confusionMatrix(predictSVM, testData$CHURN_IND)

#tuning SVM to find best cost and gamma 
  #svm_tune <- tune(svm, train.x=x, train.y=y, 
  #                kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
  #print(svm_tune)
#run the model again
  #svm_model_after_tune <- svm(Species ~ ., data=iris, kernel="radial", cost=1, gamma=0.5)
  #summary(svm_model_after_tune)
#run prediction again
  #pred <- predict(svm_model_after_tune,x)
  #system.time(predict(svm_model_after_tune,x))
#run confusion matrix again





###3. Conditional Inference Tree### Similar to DT but with extra information in the terminal nodes.
  #uses party package. A non-parametric (no assumptions) class of building regression 
  #trees, helps to overcome overfitting with exhausive breakdown procedures. #DT uses RPART.
install.packages("party")
library(party)
set.seed(123)
fit_ctree <- ctree(trainData$CHURN_IND ~ ., data=trainData)
plot(fit_ctree) #240 terminal nodes. So can't show plot

#predict on test data
predict_ctree <- predict(fit_ctree, testData)

#confusion matrix
confusionMatrix(predict_ctree, testData$CHURN_IND)





#4. Naive Bayes
library(e1071)
set.seed(700)
fit_naive <- naiveBayes (CHURN_IND ~., trainData)
fit_naive
predict_naive <- predict (fit_naive, testData)
confusionMatrix (predict_naive, testData$CHURN_IND)





###5. GBT###
library(MASS)
library(gbm)
set.seed(200)
fit_boost = gbm(CHURN_IND ~ ., trainData, distribution = "gaussian", n.trees = 10000,
                shrinkage = 0.01, interaction.depth = 4)
summary(fit_boost) 


#partial dependence plot to show relationship between churn_ind and top variables
#plot(fit_boost, i=monthly_spend_gst) 
#predict_boost <- predict(fit_boost, testData, n.trees = n.trees)
#confusionMatrix (predict_boost, testData$CHURN_IND)

  #https://datascienceplus.com/gradient-boosting-in-r/







###ANN###
#set.seed(200)
#library(neuralnet)
#  allVars <- colnames(myDataNew)
#  predictorVars <- allVars[!allVars %in% "CHURN_IND"]
#  predictorVars <- paste (predictorVars, collapse = "+")
#  form <- as.formula(paste("CHURN_IND~", predictorVars, collapse = "+"))
#fit_ANN <- neuralnet(form, trainData, hidden = 10)
#plot(fit_ANN)
#neuralnet(trainData$CHURN_IND ~., trainData, threshold = 0.01)




#Classification Tree (CART)#
#install.packages('rpart')
rpart2<- rpart(Species ~ ., data = train, method = "class")
#check variable importance
rpart2$variable.importance
printcp(rpart2)
pruned <- prune(rpart2, cp=0.05)
#predict test data using pruned model
pred <- predict(pruned, newdata=test, type="class")
#confusion matrix
table(pred, test$Species)
