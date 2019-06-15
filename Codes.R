library("dplyr")
library("DT")
library("kableExtra")
library("caret")
library("corrplot")
library("tidyverse")
library("rpart")
library("rpart.plot")
library("plotly")
library("class")
library("ROCR")
library("rms")
library(MASS)
library(randomForest)
library(xgboost)
library(pROC)
library(UBL)

w####Reading data
data <- read.csv("winequality-red.csv")

### Data Summary
summary(data)

## Box Plot
boxplot(data,col='gray',main='Wine quality values',xaxt = 'n',  xlab = '')
axis(1, labels = FALSE)
text(x =  seq_along(colnames(data)), y = par("usr")[3] - 1, srt = 45, adj = 1,labels = colnames(data), xpd = TRUE)

## Remove rows with outliers
data<- data %>%
  filter(total.sulfur.dioxide<quantile(data$total.sulfur.dioxide,.99))


### Distribution of wine quality
ggplot(data,aes(x=quality))+geom_bar(stat = "count",position = "dodge")+
  scale_x_continuous(breaks = seq(3,8,1))+
  ggtitle("Distribution of Red Wine Quality Ratings")





### Corrplot
data$quality<- as.numeric(data$quality)
data %>% cor() %>% corrplot(method = "number",type="lower")

### Alcohol,volatile.acidity,sulphates are the most significant ones

colnames(data)
###  
plot_ly(data, x=~alcohol, y=~volatile.acidity, size = ~sulphates,
        
        type="scatter",mode="markers", color = ~quality, text=~quality)


### Test And Training set
set.seed(1234)
data$quality <- as.factor(data$quality)
index<-createDataPartition(data$quality, p= .8, list=FALSE)
Train.data<-data[index,]
Test.data<-data[-index,]


### Decision Tree

model_rpart<-rpart(formula = quality ~ ., data = Train.data)
visTree(model_rpart)
rpart_result <- predict(model_rpart, newdata = Test.data[,!colnames(Test.data) %in% c("quality")],type='class')

cfrpart<-confusionMatrix(rpart_result, Test.data$quality)
1-cfrpart$overall[1]

# ### Ordinal Logistic Regression
# ologit<- polr(formula = quality ~ ., data = Train.data)
# summary(ologit)
# ##P-value calculation
# ctable<-coef(summary(ologit))
# p<-pnorm(abs(ctable[,"t value"]),lower.tail=FALSE)*2
# ctable<-cbind(ctable,"pvalue"=p)
# ctable[,4]<0.05
# 
# 
# # Final MOdel looking at P-values
# finalologit<- polr(formula = quality ~ .-citric.acid-residual.sugar-pH, data = Train.data)
# summary(finalologit)
# 
# # Confusion Matrix
# pred<-predict(finalologit,Test.data)
# (tab<-table(pred,Test.data$quality))
# # Misclassiication Rate
# 1-sum(diag(tab))/sum(tab)
# 
# names(getModelInfo())

fit_ctrl <- trainControl(method = 'cv', number = 5, classProbs = FALSE)

lr_fit <- train(
  form = quality ~ ., data = Train.data,
  method = 'polr', metric = 'Kappa',
  trControl = fit_ctrl
)

lr_prob <- predict(lr_fit, Test.data, type = 'prob')

# Confusion Matrix
pred<-predict(lr_fit,Test.data)
(tab<-table(pred,Test.data$quality))
# Misclassiication Rate
1-sum(diag(tab))/sum(tab)

############Random Forest

rf_grid <- expand.grid(mtry = c(2, 3, 5, 7))

rf_fit <- train(
  form = quality ~ ., data = Train.data,
  method = 'rf',
  trControl = fit_ctrl,
  tuneGrid = rf_grid,
  metric = 'Kappa'
)

rf_prob <- predict(rf_fit, Test.data)


# Confusion Matrix
(tab<-table(rf_prob,Test.data$quality))
# Misclassiication Rate
1-sum(diag(tab))/sum(tab)


###########KNN Classification
knn_grid <- expand.grid(k = 1:5)

knn_fit <- train(
  form = quality ~ ., data = Train.data,
  method = 'knn',
  trControl = fit_ctrl,
  preProcess = c('center', 'scale'),
  tuneGrid = knn_grid,
  metric = 'Kappa'
)

knn_prob <- predict(knn_fit, Test.data)

# Confusion Matrix
(tab<-table(knn_prob,Test.data$quality))
# Misclassiication Rate
1-sum(diag(tab))/sum(tab)




############Gradient boosting
data.train <- xgb.DMatrix(data = data.matrix(Train.data[, !colnames(Train.data) %in% c("quality")]), label = Train.data$quality)
data.valid <- xgb.DMatrix(data = data.matrix(Test.data[, !colnames(Test.data) %in% c("quality")]))



# Defining parameters of the boosting
parameters <- list(
  booster = "gbtree", silent = 0, eta = 0.08,  gamma = 0.7,
  max_depth = 8, min_child_weight = 2, subsample = .9, 
  colsample_bytree   = .5, colsample_bylevel  = 1,lambda  = 1,    
  alpha = 0,  objective   = "multi:softmax",  eval_metric = "merror",
  num_class= 7,seed= 1,tree_method = "hist", grow_policy = "lossguide"
)
###To find the best No of iterations
xgb_cv <- xgb.cv(parameters, data.train,nfold=3,nrounds = 100)

best_iter <- as.integer(which.min(xgb_cv$evaluation_log$test_merror_mean))


xgb_model <- xgb.train(parameters, data.train, nrounds = best_iter)
xgb_prob <- predict(xgb_model, data.valid)
# Confusion Matrix
cm<-confusionMatrix(as.factor(xgb_prob+2), Test.data$quality)
#Misclassification Rate
1-cm$overall[1]


######Neural Network

nn_grid <-expand.grid(size = seq(from = 1, to = 5, by = 1),
                                  decay = seq(from = 0.1, to = 0.5, by = 0.1))


nn_fit <- train(
  form = quality ~ ., data = Train.data,
  method = 'nnet',
  trControl = fit_ctrl,
  tuneGrid =nn_grid,
  metric = 'Kappa'
)

nn_prob <- predict(nn_fit, Test.data)
# Confusion Matrix
(tab<-table(nn_prob,Test.data$quality))
# Misclassiication Rate
1-sum(diag(tab))/sum(tab)




#####Best model is Random Forest,so some analysis oN RF model
rf_model <- randomForest(quality~.,Train.data)

rf_result <- predict(rf_model, newdata = Test.data[,!colnames(Test.data) %in% c("quality")])


confusionMatrix(rf_result, Test.data$quality)
##Finding best predictors

varImpPlot(rf_model)

####AUC Curve
auc<-multiclass.roc(as.numeric(Test.data$quality),as.numeric(rf_result))
print(auc)



###To treat imbalanced data
### Can be skipped from the report
trainSplit<-SmoteClassif(quality~.,Train.data,C.perc = "balance",k=3,repl=FALSE,dist = "Euclidean")
testSplit<-SmoteClassif(quality~.,Test.data,C.perc = "balance",k=1,repl=FALSE,dist = "Euclidean")
table(testSplit$quality)


rf_smote_model <- randomForest(quality~.,trainSplit)

rf_smote_result <- predict(rf_model, newdata = testSplit[,!colnames(testSplit) %in% c("quality")])

confusionMatrix(rf_smote_result, testSplit$quality)

auc_smote<-multiclass.roc(as.numeric(testSplit$quality),as.numeric(rf_smote_result))
print(auc_smote)


