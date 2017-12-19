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

lr = [0.06 0.1 0.5 1];
maxNumSplits = 6;
minleafsize = 10;
numTrees = [500 1000];
t = templateTree('MaxNumSplits',maxNumSplits,'Surrogate','on');
subsample = [0.5 0.6 0.8 1];
n_s = numel(subsample);
MSE = zeros(4,4,2);
for k = 1:length(numTrees)
    for i = 1:length(lr)
        for j = 1:n_s     
            Mdl = fitrensemble(X_train,y_train,'NumLearningCycles',numTrees(k),'Learners',t,...
                'LearnRate',lr(i),'Resample','on','FResample',subsample(j),...
                'Replace','off','KFold',5);
            MSE(i,j,k) = kfoldLoss(Mdl);
        end
    end
end

hold on
plot(subsample,MSE(1,:,1),'-*','LineWidth',1)
plot(subsample,MSE(2,:,1),'-*','LineWidth',1)
plot(subsample,MSE(3,:,1),'-*','LineWidth',1)
plot(subsample,MSE(4,:,1),'-*','LineWidth',1)
plot(subsample,MSE(1,:,2),'-o','LineWidth',1)
plot(subsample,MSE(2,:,2),'-o','LineWidth',1)
plot(subsample,MSE(3,:,2),'-o','LineWidth',1)
plot(subsample,MSE(4,:,2),'-o','LineWidth',1)
xlabel('Subsample Fraction');
ylabel('5-Fold CV MSE');
legend('lr=0.06,ntrees=500','lr=0.1,ntrees=500','lr=0.5,ntrees=500','lr=1,ntrees=500',...
     'lr=0.06,ntrees=1000','lr=0.1,ntrees=1000','lr=0.5,ntrees=1000','lr=1,ntrees=1000')
grid
figure
hold on
plot(lr,MSE(:,1,1),'-*','LineWidth',1)
plot(lr,MSE(:,2,1),'-*','LineWidth',1)
plot(lr,MSE(:,3,1),'-*','LineWidth',1)
plot(lr,MSE(:,4,1),'-*','LineWidth',1)
plot(lr,MSE(:,1,2),'-o','LineWidth',1)
plot(lr,MSE(:,2,2),'-o','LineWidth',1)
plot(lr,MSE(:,3,2),'-o','LineWidth',1)
plot(lr,MSE(:,4,2),'-o','LineWidth',1)
xlabel('Learning Rate');
ylabel('5-Fold CV MSE');
legend('f=0.5,ntrees=500','f=0.6,ntrees=500','f=0.8,ntrees=500','f=1,ntrees=500',...
     'f=0.5,ntrees=1000','f=0.6,ntrees=1000','f=0.8,ntrees=1000','f=1,ntrees=1000')
 grid
 
[r1, c1]=find(MSE(:,:,1)==min(min(MSE(:,:,1))))
[r2, c2]=find(MSE(:,:,2)==min(min(MSE(:,:,2))))