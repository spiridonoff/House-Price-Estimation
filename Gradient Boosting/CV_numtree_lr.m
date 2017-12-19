clc
clear

X_train = importdata('X_train_scaled.csv');
X_train = X_train.data;
y_train = importdata('y_train_scaled.csv');
y_train = y_train.data;
X_test = importdata('X_test_scaled.csv');
X_test = X_test.data;
y_test = importdata('y_test_scaled.csv');
y_test = y_test.data;

lr = [0.06 0.1 0.3 0.5 1];
n_lr = numel(lr);
maxNumSplits = 6;
numTrees = [200 500 1000];
t = templateTree('MaxNumSplits',maxNumSplits,'Surrogate','on');
MSE = zeros(size(numTrees,2),n_lr);

for i = 1:n_lr  
    for j = 1:length(numTrees)
        Mdl = fitrensemble(X_train,y_train,'NumLearningCycles',numTrees(j),'Learners',...
            t,'LearnRate',lr(i),'KFold',5);
        MSE(j,i) = kfoldLoss(Mdl);
    end
end
plot(lr,MSE(1,:),'.-',lr,MSE(2,:),'.-',lr,MSE(3,:),'.-')
xlabel('LearnRate');
ylabel('5-Fold CV MSE');
title('Cross-Validation of Number of Trees and Learn Rate')
legend('ntrees=200','ntrees=500','ntrees=1000');
grid
