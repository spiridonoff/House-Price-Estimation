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
numTrees = 500;
t = templateTree('MaxNumSplits',maxNumSplits,'Surrogate','on');

MSE_train = zeros(n_lr,1);
MSE_test = zeros(n_lr,1);
MAE_train = zeros(n_lr,1);
MAE_test = zeros(n_lr,1);
mu_ytrain = mean(y_train);
mu_ytest = mean(y_test);
SStot_train = sum((y_train - mu_ytrain).^2);
SStot_test = sum((y_test - mu_ytest).^2);
R2_train = zeros(n_lr,1);
R2_test = zeros(n_lr,1);
for r = 1:length(lr)  
    Mdl = fitensemble(X_train,y_train,'LSBoost',numTrees,t,...
        'Type','regression','LearnRate',lr(r));
    ypred_train = predict(Mdl,X_train);
    ypred_test = predict(Mdl,X_test);
    MAE_train(r) = mean(abs(y_train-ypred_train));
    MAE_test(r) = mean(abs(y_test-ypred_test));
    MSE_train(r) = mean(abs(y_train-ypred_train).^2);
    MSE_test(r) = mean(abs(y_test-ypred_test).^2);
    R2_train(r) = 1-(sum((y_train-ypred_train).^2)/SStot_train);
    R2_test(r) = 1-(sum((y_test-ypred_test).^2)/SStot_test);
end
subplot(2,2,1);
plot(lr,MAE_train,':',lr,MAE_test,'--');
legend('training data','test data');
xlabel('learning rate');
ylabel('MAE');
grid
subplot(2,2,2);
plot(lr,MSE_train,':',lr,MSE_test,'--');
legend('training data','test data');
xlabel('learning rate');
ylabel('MSE');
grid
subplot(2,2,3);
plot(lr,sqrt(MSE_train),':',lr,sqrt(MSE_test),'--');
legend('training data','test data');
xlabel('learning rate');
ylabel('RMSE');
grid
subplot(2,2,4);
plot(lr,R2_train,':',lr,R2_test,'--');
legend('training data','test data');
xlabel('learning rate');
ylabel('R^2');
grid
