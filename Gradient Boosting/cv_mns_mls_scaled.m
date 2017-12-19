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

numTrees = 150;
minleafsize = [2 3 6 8 15 25];
maxnumsplits =[2 3 6 8 15 25];

MSE = zeros(6);
for i = 1:length(minleafsize)
    for j = 1:length(maxnumsplits)
        t = templateTree('MaxNumSplits',maxnumsplits(j),'MinLeafSize',minleafsize(i),'Surrogate','on');
        Mdl = fitrensemble(X_train,y_train,'NumLearningCycles',numTrees,'Learners',t,'KFold',5);
        MSE(i,j) = kfoldLoss(Mdl);
    end
end

plot(maxnumsplits,MSE(1,:),'.-',maxnumsplits,MSE(2,:),'.-',maxnumsplits,MSE(3,:),'.-',...
maxnumsplits,MSE(4,:),'.-',maxnumsplits,MSE(5,:),'.-',maxnumsplits,MSE(6,:),'.-')
legend('MLS=2','MLS=3','MLS=6','MLS=8','MLS=15','MLS=25')
xlabel('Max. Number of Splits')
ylabel('5-fold cross validated MSE')
title('CV MSE for different values of MinLeafSize and MaxNumSplits') 
grid
plot(minleafsize,MSE(:,1),'.-',minleafsize,MSE(:,2),'.-',minleafsize,MSE(:,3),'.-',...
minleafsize,MSE(:,4),'.-',minleafsize,MSE(:,5),'.-',minleafsize,MSE(:,6),'.-')
legend('MNS=2','MNS=3','MNS=6','MNS=8','MNS=15','MNS=25')
xlabel('Min. Leaf Size')
ylabel('5-fold cross validated MSE')
title('CV MSE for different values of MinLeafSize and MaxNumSplits') 
grid

%{
rng(1); % For reproducibility
t = templateTree('MaxNumSplits',1);
Mdl1 = fitrensemble(X_train,y_train,'NumLearningCycles',500,'Learners',t,'CrossVal','on');
kflc = kfoldLoss(Mdl1,'Mode','cumulative');
figure;
plot(kflc);
ylabel('10-fold cross-validated MSE');
xlabel('Learning cycle');
%}