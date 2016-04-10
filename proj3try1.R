# Author: Xin Wang
# Email: xinwangmath@gmail.com

# Extra experiment
# Based on observation for neural network, we set number of hidden nodes to be 2 and decay to be 0.5

myData = read.csv("AngleClosure.csv", header = TRUE, na.strings = "NA")

removeFeatures = c("EYE", "GENDER", "ETHNIC", "HGT", "WT", "ASPH", "ACYL", "SE", "AXL", "CACD", "AGE", "CCT.OD","PCCURV_mm")
removeIndices = rep(0, length(removeFeatures)); 
for(tempInd in 1:length(removeFeatures) ){
	removeIndices[tempInd] = which(attributes(myData)$names == removeFeatures[tempInd])
}
myData = myData[, -removeIndices]

myLogical = apply(myData, 1, function(xx){
	return( !any(is.na(xx)) ); 
	}) 
myData = myData[myLogical, ]

# cast predictors into numeric variables and response as a categorical variable with level 1 and 0
responseIndex = which(attributes(myData)$names == "ANGLE.CLOSURE")
myPredictors = data.matrix(myData[, -responseIndex])
ANGLE.CLOSURE = factor(sapply(myData[, responseIndex], function(xx){
	return( ifelse(xx == "YES", 1, 0))
	}))
#ANGLE.CLOSURE = myData[, responseIndex]
dataStar = data.frame(ANGLE.CLOSURE, myPredictors)

library(nnet)
library(pROC)

nn_nodes_opt = 2
nn_decay_opt = 0.5
nn_fit_opt = nnet(ANGLE.CLOSURE~., data = dataStar, size = nn_nodes_opt, 
	decay = nn_decay_opt, maxit = 250)



valDataCase = read.csv("AngleClosure_ValidationCases.csv", header = TRUE, na.strings = "NA")
valDataControl = read.csv("AngleClosure_ValidationControls.csv", header = TRUE, na.strings = "NA")

valCase_names = attributes(valDataCase)$names 
valControl_names = attributes(valDataControl)$names

model_features = attributes(dataStar)$names[-1]
# get the relevant features from case and control data
# unbelievable, this works!
source("selectFeatures.R")
case_names = selectFeatures(valCase_names, model_features)
control_names = selectFeatures(valControl_names, model_features, "\\.")

# only keep the relevant features
# delete rows with any missing data
valDataCase = valDataCase[, case_names]
myLogical = apply(valDataCase, 1, function(xx){
	return( !any(is.na(xx)) ); 
	}) 
valDataCase = valDataCase[myLogical, ]

valDataControl = valDataControl[, control_names]
myLogical = apply(valDataControl, 1, function(xx){
	return( !any(is.na(xx)) ); 
	}) 
valDataControl = valDataControl[myLogical, ]

case_predictors = data.matrix(valDataCase)
control_predictors = data.matrix(valDataControl)

case_response = rep(1, dim(valDataCase)[1])
control_response = rep(0, dim(valDataControl)[1])

ANGLE.CLOSURE = factor(sapply(case_response, function(xx){
	return( ifelse(xx == 1, 1, 0))
	}))
case_data = data.frame(ANGLE.CLOSURE, case_predictors)
attributes(case_data)$names = attributes(dataStar)$names

ANGLE.CLOSURE = factor(sapply(control_response, function(xx){
	return( ifelse(xx == 1, 1, 0))
	}))
control_data = data.frame(ANGLE.CLOSURE, control_predictors)
attributes(control_data)$names = attributes(dataStar)$names

# all the case and control data are combined into the val_data data frame
val_data = rbind(case_data, control_data)

nn_val_preds = predict(nn_fit_opt, newdata = val_data)
nn_val_roc = roc(val_data$ANGLE.CLOSURE, nn_val_preds)
nn_val_auc = auc(nn_val_roc)
# val_auc[3] = nn_val_auc

pdf(file = "nn_try.pdf", width = 5, height = 6)
plot(nn_val_roc)
dev.off()
print(nn_val_auc)



