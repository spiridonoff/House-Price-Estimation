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

numTrees = 500;

MSE = zeros(25);
for i = 1:25
    for j = 1:25
        t = templateTree('MaxNumSplits',j,'MinLeafSize',i,'Surrogate','on');
        Mdl = fitrensemble(X_train,y_train,'NumLearningCycles',numTrees,'Learners',t,'KFold',5);
        MSE(i,j) = kfoldLoss(Mdl);
    end
end

plot(1:25,MSE(:,1),1:25,MSE(:,2),1:25,MSE(:,6),1:25,MSE(:,8),1:25,MSE(:,16),1:25,MSE(:,25))
legend('MNS=1','MNS=2','MNS=6','MNS=8','MNS=16','MNS=25')
xlabel('Min. Leaf Size')
ylabel('10-fold cross validated MSE')
title('Loss as function of Min. Leaf Size and Max. Number of Splits') 
grid
plot(1:25,MSE(1,:),1:25,MSE(2,:),1:25,MSE(6,:),1:25,MSE(8,:),1:25,MSE(16,:),1:25,MSE(25,:))
legend('MLS=2','MLS=3','MLS=6','MLS=8','MLS=16','MLS=25')
xlabel('Max. Number of Splits')
ylabel('5-fold cross validated MSE')
title('CV MSE for different values of MinLeafSize and MaxNumSplits') 
grid
