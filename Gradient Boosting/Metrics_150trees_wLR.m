clc
clear

X_train = importdata('X_train.csv');
X_train = X_train.data;
y_train = importdata('y_train.csv');
y_train = y_train.data;
X_test = importdata('X_test.csv');
X_test = X_test.data;
y_test = importdata('y_test.csv');
y_test = y_test.data;

maxNumSplits = 6;
tree = templateTree('MaxNumSplits',maxNumSplits,'Surrogate','on');
MSE_train = zeros(150,1);
MSE_test = zeros(150,1);
MAE_train = zeros(150,1);
MAE_test = zeros(150,1);
mu_ytrain = mean(y_train);
mu_ytest = mean(y_test);
SStot_train = sum((y_train - mu_ytrain).^2);
SStot_test = sum((y_test - mu_ytest).^2);
R2_train = zeros(150,1);
R2_test = zeros(150,1);
for t = 1:150
    Mdl = fitrensemble(X_train,y_train,'NumLearningCycles',t,'Learners',tree,'LearnRate',0.1);
    ypred_train = predict(Mdl,X_train);
    ypred_test = predict(Mdl,X_test);
    MAE_train(t) = mean(abs(y_train-ypred_train));
    MAE_test(t) = mean(abs(y_test-ypred_test));
    MSE_train(t) = mean(abs(y_train-ypred_train).^2);
    MSE_test(t) = mean(abs(y_test-ypred_test).^2);
    R2_train(t) = 1-(sum((y_train-ypred_train).^2)/SStot_train);
    R2_test(t) = 1-(sum((y_test-ypred_test).^2)/SStot_test);
end
numcycles = 1:150;
subplot(2,2,1);
plot(numcycles,MAE_train,':',numcycles,MAE_test,'--');
legend('training data','test data');
xlabel('Number of Cycles');
ylabel('MAE');
grid
subplot(2,2,2);
plot(numcycles,MSE_train,':',numcycles,MSE_test,'--');
legend('training data','test data');
xlabel('Number of Cycles');
ylabel('MSE');
grid
subplot(2,2,3);
plot(numcycles,sqrt(MSE_train),':',numcycles,sqrt(MSE_test),'--');
legend('training data','test data');
xlabel('Number of Cycles');
ylabel('RMSE');
grid
subplot(2,2,4);
plot(numcycles,R2_train,':',numcycles,R2_test,'--');
legend('training data','test data','Location','southeast');
xlabel('Number of Cycles');
ylabel('R^2');
grid
