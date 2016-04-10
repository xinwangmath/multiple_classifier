# Author: Xin Wang
# Email: xinwangmath@gmail.com


# step-1 Data manipulation
#  (a) read in data; (b) delete the columns corresponding to EYE, GENDER, ETHNIC; 
#  (c) delte rows with any missing values

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

# step-2. Prediction models
#  we use 1. logistic regression, 2. random forest, 3. neural network, 4. AdaBoost, 5 SVM. 
#   actively altered tuning parameters: 
#     1. none
#     2. the number of variables to use in each step, we choose mtry: from /sqrt(p) to all varaibles
#     3. decay, (and number of hidden nodes: 5, 10, 15, 20)
#     4. mfinal(number of trees) from small to large, (and maxdepth = 5 6 7 8)
#     5. if use kernel = "linear", then choose cost; 
#        if use kernel = "radial", choose cost and gamma (gamma = 0.5, 1, 2)

library(randomForest)
library(nnet)
library(adabag)
library(e1071)

library(pROC)


# step-3. cross-validation

totalPts = dim(dataStar)[1]
num_features = dim(dataStar)[2] - 1

nFolds = 10
nIter = 10

glm_auc = matrix(NA, nIter, 1)
glm_records = list()

rf_mtry = seq(round(sqrt(num_features)), num_features, 1)
rf_auc = matrix(NA, nIter, length(rf_mtry))
rf_records = list()
rf_records_mat = matrix(NA, round(totalPts/nFolds), length(rf_mtry))

nn_decay = 10^seq(-1, 5, 1)
nn_nodes = c(5, 10, 15, 20)
nn_auc = matrix(NA, nIter, (length(nn_decay) * length(nn_nodes)) )
nn_records = list()
nn_records_mat = matrix(NA, round(totalPts/nFolds), (length(nn_decay) * length(nn_nodes)) )

boost_mfinal = c(1, 50, 100, 200)
boost_maxdepth = c(5, 6, 7, 8)
boost_auc = matrix(NA, nIter, (length(boost_maxdepth) * length(boost_mfinal)) )
boost_records = list()
boost_records_mat = matrix(NA, round(totalPts/nFolds), (length(boost_maxdepth) * length(boost_mfinal)) )

svm_gamma = c(0.5, 1, 2)
svm_cost = c(0.5, 1, 5, 10)
svm_auc = matrix(NA, nIter, (length(svm_gamma) * length(svm_cost)) )
svm_records = list()
svm_records_mat = matrix(NA, round(totalPts/nFolds), (length(svm_gamma) * length(svm_cost)) )

testing_true_values = list()




set.seed(1)

for(iter in 1:nIter){
	testingIndices = sample(totalPts)[1:round(totalPts/nFolds)]
	myDataTesting = dataStar[testingIndices, ]
	myDataTraining = dataStar[-testingIndices, ]

	print(iter)

	# 1. logistic regression
	glm_fit_temp = glm(ANGLE.CLOSURE~., data = myDataTraining, family = binomial)
	glm_fit = step(glm_fit_temp, k = log(length(testingIndices)), trace = 0)
	glm_preds = predict(glm_fit, newdata = myDataTesting, type = "response")
	# glm_preds stores the probability for the predition to be 1

	# compute the corresponding auc
	glm_roc = roc(myDataTesting$ANGLE.CLOSURE, glm_preds)
	glm_auc[iter] = auc(glm_roc)

	print("glm")

	# record the predictions for this testing data set
	glm_records[[iter]] = glm_preds




	# 2. random forest
	for(rf_m in rf_mtry){
		rf_fit = randomForest(ANGLE.CLOSURE~., data = myDataTraining, mtry = rf_m)
		rf_results = predict(rf_fit, newdata = myDataTesting, type = "prob")
		rf_preds = rf_results[, 2]
		# rf_preds is the probability for the prediction to be 1

		rf_roc = roc(myDataTesting$ANGLE.CLOSURE, rf_preds)
		rf_auc[iter, rf_mtry == rf_m] = auc(rf_roc)

		# record the preds
		rf_records_mat[, rf_mtry == rf_m] = rf_preds
	}

	print("rf")

	# record the predictions for this testing data set
	rf_records[[iter]] = rf_records_mat


	# 3. neural network
	for(ii in 1:length(nn_nodes)){
		for(jj in 1:length(nn_decay)){

			nn_fit = nnet(ANGLE.CLOSURE~., data = myDataTraining, size = nn_nodes[ii], 
				decay = nn_decay[jj], maxit = 250)
			nn_preds = predict(nn_fit, newdata = myDataTesting)
			# nn_preds is the probability for the prediction to be 1

			nn_roc = roc(myDataTesting$ANGLE.CLOSURE, nn_preds)
			nn_auc[iter, ((ii - 1) * length(nn_decay) + jj)] = auc(nn_roc)

			# record the preds
			nn_records_mat[,((ii - 1) * length(nn_decay) + jj)] = nn_preds
		}
	}

	print("nn")

	# record the predictions for this tesing data set
	nn_records[[iter]] = nn_records_mat


	# 4. ada boost
	# version 1: may run slow, in that case, swith to version 2
	for(ii in 1:length(boost_maxdepth)){
		for(jj in 1:length(boost_mfinal)){

			boost_fit = boosting(ANGLE.CLOSURE~., data = myDataTraining, mfinal = boost_mfinal[jj], 
				coeflearn = "Freund", control = rpart.control(maxdepth = boost_maxdepth[ii]))
			boost_results = predict(boost_fit, newdata = myDataTesting)
			boost_preds = boost_results$prob[, 2]
			# boost_preds is the probability for the prediction to be 1

			boost_roc = roc(myDataTesting$ANGLE.CLOSURE, boost_preds)
			boost_auc[iter, ((ii-1) * length(boost_mfinal) + jj)] = auc(boost_roc)

			# record the preds
			boost_records_mat[, ((ii-1) * length(boost_mfinal) + jj)] = boost_preds
		}
	}

	print("boost")

	# record the predictions for this testing data set
	boost_records[[iter]] = boost_records_mat


	# 5. SVM: note by default, variables are scaled
	for(ii in 1:length(svm_gamma)){
		for(jj in 1:length(svm_cost)){

			svm_fit = svm(ANGLE.CLOSURE~., data = myDataTraining, kernel = "radial", 
				gamma = svm_gamma[ii], cost = svm_cost[jj], probability = TRUE)
			svm_results = predict(svm_fit, newdata = myDataTesting, probability = TRUE)
			svm_preds = attr(svm_results, "probabilities")[, "1"]
			# svm_preds is the probability for the prediction to be 1

		    svm_roc = roc(myDataTesting$ANGLE.CLOSURE, svm_preds)
			svm_auc[iter, ((ii-1) * length(svm_cost) + jj)] = auc(svm_roc)

			# record the preds
			svm_records_mat[, ((ii-1) * length(svm_cost) + jj)] = svm_preds
		}
	}

	print("svm")

	# record the predictions for this testing data set
	svm_records[[iter]] = svm_records_mat


	# record the true response value for this testing data set
	testing_true_values[[iter]] = myDataTesting$ANGLE.CLOSURE

}

glm_auc_ave = mean(glm_auc)
rf_auc_ave = apply(rf_auc, 2, mean)
nn_auc_ave = apply(nn_auc, 2, mean)
boost_auc_ave = apply(boost_auc, 2, mean)
svm_auc_ave = apply(svm_auc, 2, mean)

# find the optimal parameters and print out them and the AUC
print(glm_auc_ave)

rf_opt = which.max(rf_auc_ave)
rf_mtry_opt = rf_mtry[rf_opt]
print(rf_mtry_opt)
print(rf_auc_ave[rf_opt])

nn_opt = which.max(nn_auc_ave)
if(nn_opt %% length(nn_decay) == 0){
	nn_nodes_opt = nn_nodes[nn_opt %/% length(nn_decay)]
} else{
	nn_nodes_opt = nn_nodes[nn_opt %/% length(nn_decay) + 1]
}
temp_ind = ifelse(nn_opt %% length(nn_decay) > 0, nn_opt %% length(nn_decay), length(nn_decay))
nn_decay_opt = nn_decay[temp_ind]
print(nn_nodes_opt)
print(nn_decay_opt)
print(nn_auc_ave[nn_opt])

boost_opt = which.max(boost_auc_ave)
if(boost_opt %% length(boost_mfinal) == 0){
	boost_maxdepth_opt = boost_maxdepth[boost_opt %/% length(boost_mfinal)]
} else{
	boost_maxdepth_opt = boost_maxdepth[boost_opt %/% length(boost_mfinal) + 1]
}
temp_ind = ifelse(boost_opt %% length(boost_mfinal) > 0, boost_opt %% length(boost_mfinal), length(boost_mfinal))
boost_mfinal_opt = boost_mfinal[temp_ind]
print(boost_maxdepth_opt)
print(boost_mfinal_opt)
print(boost_auc_ave[boost_opt])

svm_opt = which.max(svm_auc_ave)
if(svm_opt %% length(svm_cost) == 0){
	svm_gamma_opt = svm_gamma[svm_opt %/% length(svm_cost)]
} else{
	svm_gamma_opt = svm_gamma[svm_opt %/% length(svm_cost) + 1]
}
temp_ind = ifelse(svm_opt %% length(svm_cost) > 0, svm_opt %% length(svm_cost), length(svm_cost))
svm_cost_opt = svm_cost[temp_ind]
print(svm_gamma_opt)
print(svm_cost_opt)
print(svm_auc_ave[svm_opt])

# 4. Stacking
# We use 0-1 to code "NO" and "YES"
# step-1. use the optimal parameters to train the models, stacking the 10 cv tesing results
#         together to form the y vector and the five x vectors
# step-2. use OLS to find the first set of coefficents
# step-3. use quadprog to find the second set of coefficients

myY = rep(NA, round(totalPts/nFolds) * nFolds)
myX = matrix(NA, round(totalPts/nFolds) * nFolds, 5)
colnames(myX) = c("glm", "rf", "nn", "boost", "svm")


for(iter in 1:nIter){
	

	myY[ ((iter-1) * round(totalPts/nFolds) + 1) : (iter * round(totalPts/nFolds)) ] = testing_true_values[[iter]]
	myX[ ((iter-1) * round(totalPts/nFolds) + 1) : (iter * round(totalPts/nFolds)) , "glm"] = glm_records[[iter]]
	myX[ ((iter-1) * round(totalPts/nFolds) + 1) : (iter * round(totalPts/nFolds)) , "rf"] = rf_records[[iter]][, rf_opt]
	myX[ ((iter-1) * round(totalPts/nFolds) + 1) : (iter * round(totalPts/nFolds)) , "nn"] = nn_records[[iter]][, nn_opt]
	myX[ ((iter-1) * round(totalPts/nFolds) + 1) : (iter * round(totalPts/nFolds)) , "boost"] = boost_records[[iter]][, boost_opt]
	myX[ ((iter-1) * round(totalPts/nFolds) + 1) : (iter * round(totalPts/nFolds)) , "svm"] = svm_records[[iter]][, svm_opt]

}

# unconstrained stacking

#w1 = solve( t(myX) %*% myX, t(myX) %*% myY)

# or 
simple_stacking = lm(myY~myX-1)
w1 = coef(simple_stacking)

library(quadprog)

Dmat = t(myX) %*% myX
dvec = t(myY) %*% myX
Amat = matrix(0, 5, 6)
Amat[, 1] = rep(1, 5)
Amat[1, 2] = 1
Amat[2, 3] = 1
Amat[3, 4] = 1
Amat[4, 5] = 1
Amat[5, 6] = 1
bvec = c(1, 0, 0, 0, 0, 0)
meq = 1

w2 = solve.QP(Dmat = Dmat, dvec = dvec, Amat = Amat, bvec = bvec, meq = 1)$solution


# Train the 5 base models using the whole data set and the optimal parameters
# For the two stacking models, we don't need to do this. The models are already given by w1 and w2

# 1. logistic regression

glm_fit_temp = glm(ANGLE.CLOSURE~., data = dataStar, family = binomial)
glm_fit_opt = step(glm_fit_temp, k = log(totalPts), trace = 0)

# 2. random forest
rf_fit_opt = randomForest(ANGLE.CLOSURE~., data = dataStar, mtry = rf_mtry_opt)

# 3. neural network
nn_fit_opt = nnet(ANGLE.CLOSURE~., data = dataStar, size = nn_nodes_opt, 
	decay = nn_decay_opt, maxit = 250)

# 4. AdaBoost
boost_fit_opt = boosting(ANGLE.CLOSURE~., data = dataStar, mfinal = boost_mfinal_opt, 
		coeflearn = "Freund", control = rpart.control(maxdepth = boost_maxdepth_opt))

# 5. SVM
svm_fit_opt = svm(ANGLE.CLOSURE~., data = dataStar, kernel = "radial", 
		gamma = svm_gamma_opt, cost = svm_cost_opt, probability = TRUE)


# 5. Validation
# step-1: read in the 2 validation data set, delete the columns corresponding to EYE, GENDER, ETHNIC; 
#  ,and delte rows with any missing values; 
# step-2: concatenate the two data set into one; 
# step-3: use same names for predictors and response variables
# step-4: use the optimal trained models to predict on those data sets, and compute ROC curve and AUC
# step-5: find the largest AUC

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


# find the predict, roc and auc of each model
val_auc = rep(0, 7)
# 1. glm
glm_val_preds = predict(glm_fit_opt, newdata = val_data, type = "response")
glm_val_roc = roc(val_data$ANGLE.CLOSURE, glm_val_preds)
glm_val_auc = auc(glm_val_roc)
val_auc[1] = glm_val_auc

# 2. rf
rf_val_results = predict(rf_fit_opt, newdata = val_data, type = "prob")
rf_val_preds = rf_val_results[, 2]
rf_val_roc = roc(val_data$ANGLE.CLOSURE, rf_val_preds)
rf_val_auc = auc(rf_val_roc)
val_auc[2] = rf_val_auc


# 3. nn
nn_val_preds = predict(nn_fit_opt, newdata = val_data)
nn_val_roc = roc(val_data$ANGLE.CLOSURE, nn_val_preds)
nn_val_auc = auc(nn_val_roc)
val_auc[3] = nn_val_auc


# 4. boost
boost_val_results = predict(boost_fit_opt, newdata = val_data)
boost_val_preds = boost_val_results$prob[, 2]
boost_val_roc = roc(val_data$ANGLE.CLOSURE, boost_val_preds)
boost_val_auc = auc(boost_val_roc)
val_auc[4] = boost_val_auc


# 5. SVM
svm_val_results = predict(svm_fit_opt, newdata = val_data, probability = TRUE)
svm_val_preds = attr(svm_val_results, "probabilities")[, "1"]
svm_val_roc = roc(val_data$ANGLE.CLOSURE, svm_val_preds)
svm_val_auc = auc(svm_val_roc)
val_auc[5] = svm_val_auc


# 6. simple stacking model 
models_mat = cbind(glm_val_preds, rf_val_preds, nn_val_preds, boost_val_preds, svm_val_preds)
stacking_1_preds = models_mat %*% w1
stacking_1_roc = roc(val_data$ANGLE.CLOSURE, stacking_1_preds)
stacking_1_auc = auc(stacking_1_roc)
val_auc[6] = stacking_1_auc


# 7. constrained stacking model
stacking_2_preds = models_mat %*% w2
stacking_2_roc = roc(val_data$ANGLE.CLOSURE, stacking_2_preds)
stacking_2_auc = auc(stacking_2_roc)
val_auc[7] = stacking_2_auc


# 5.2 result demonstration
# print out the AUC for each model

print("linear regression")
print(glm_val_auc)

print("Random forest")
print(rf_val_auc)

print("neural network")
print(nn_val_auc)

print("AdaBoot")
print(boost_val_auc)

print("SVM")
print(svm_val_auc)

print("simple stacking")
print(stacking_1_auc)

print("constrained stacking")
print(stacking_2_auc)


# 6. Visualization
# goals: 
#  1. for the 5 base models, plot cv AUC vs tuning parameters
#  2. for the all 7 models, plot the ROC curve on the validation data set, annotated with AUCs

# 1. AUC curves
# glm
print(glm_auc_ave)

# rf
#dev.new(width = 5, height = 6)
pdf(file = "rf", width = 5, height = 6)
plot(rf_mtry, rf_auc_ave, pch = 1)
lines(rf_mtry, rf_auc_ave)
dev.off()

# nn: two active parameters nn_nodes and nn_decay
pdf(file = "nn", width = 5, height = 6)
par(mfrow = c(2, 2))
for(ii in 1:length(nn_nodes)){
	plot(nn_decay, nn_auc_ave[((ii-1) * length(nn_decay) + 1) : (ii * length(nn_decay))], pch = 1, 
		ylab = "AUC")
	lines(nn_decay, nn_auc_ave[((ii-1) * length(nn_decay) + 1) : (ii * length(nn_decay))])
}
dev.off()

# boost: two active parameters: boost_maxdepth and boost_mfinal
pdf(file = "boost", width = 5, height = 6)
par(mfrow = c(2, 2))
for(ii in 1:length(boost_maxdepth)){
	plot(boost_mfinal, boost_auc_ave[((ii-1) * length(boost_mfinal) + 1) : (ii * length(boost_mfinal))], pch = 1, 
		ylab = "AUC")
	lines(boost_mfinal, boost_auc_ave[((ii-1) * length(boost_mfinal) + 1) : (ii * length(boost_mfinal))])
}
dev.off()

# svm: two active parameters: svm_gamma and svm_cost
pdf(file = "svm", width = 3, height = 6)
par(mfrow = c(3, 1))
for(ii in 1:length(svm_gamma)){
	plot(svm_cost, svm_auc_ave[((ii-1) * length(svm_cost) + 1) : (ii * length(svm_cost))], pch = 1, 
		ylab = "AUC")
	lines(svm_cost, svm_auc_ave[((ii-1) * length(svm_cost) + 1) : (ii * length(svm_cost))])
}
dev.off()



# 2 ROC curves
#dev.new(width = 5, height = 6)
pdf("glm_roc.pdf", width = 6, height = 6)
plot(glm_val_roc)
dev.off()

#dev.new(width = 5, height = 6)
pdf("rf_roc.pdf", width = 5, height = 6)
plot(rf_val_roc)
dev.off()

#dev.new(width = 5, height = 6)
pdf("nn_roc.pdf", width = 5, height = 6)
plot(nn_val_roc)
dev.off()

#dev.new(width = 5, height = 6)
pdf("boost_roc.pdf", width = 5, height = 6)
plot(boost_val_roc)
dev.off()

#dev.new(width = 5, height = 6)
pdf("svm_roc.pdf", width = 5, height = 6)
plot(svm_val_roc)
dev.off()

#dev.new(width = 5, height = 6)
pdf("s1_roc.pdf", width = 5, height = 6)
plot(stacking_1_roc)
dev.off()

#dev.new(width = 5, height = 6)
pdf("s2_roc.pdf", width = 5, height = 6)
plot(stacking_2_roc)
dev.off()






















































	




























