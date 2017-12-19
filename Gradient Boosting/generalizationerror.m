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

rng(1); % For reproducibility
t = templateTree('MaxNumSplits',1);
Mdl1 = fitrensemble(X_train,y_train,'NumLearningCycles',500,'Learners',t,'CrossVal','on');
kflc = kfoldLoss(Mdl1,'Mode','cumulative');
figure;
plot(kflc);
ylabel('10-fold cross-validated MSE');
xlabel('Learning cycle');