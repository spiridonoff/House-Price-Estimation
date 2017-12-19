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

%% Conditional Partial Dependence Plots
X = [X_train(:,2),X_train(:,3),X_train(:,7),X_train(:,9)];
Mdl = fitrensemble(X,y_train,...
    'PredictorNames',{'Bathrooms','Sqft_living','Grade','View'}, ...
    'ResponseName','Price');
Xnew = [X_test(:,2),X_test(:,3),X_test(:,7),X_test(:,9)];
f = figure;
f.Position = [100 100 1.75*f.Position(3:4)]; % Enlarge figure for visibility.

for i = 1 : 4
    subplot(2,2,i)
    plotPartialDependence(Mdl,i,Xnew,'Conditional','absolute');
end
%% Two-Variable Dependence Plots
X2 = [X_train(:,1),X_train(:,2),X_train(:,3),...
    X_train(:,4),X_train(:,9),X_train(:,14)];
Mdl2 = fitrensemble(X2,y_train,...
    'PredictorNames',{'Bedrooms','Bathrooms','SqftLiving','SqftLot',...
    'View','Life'},'ResponseName','Price');
figure;
pt1 = linspace(min(X2(:,3)),max(X2(:,3)),50)';
pt2 = linspace(min(X2(:,5)),max(X2(:,5)),50)';
ax1 = plotPartialDependence(Mdl2,{'SqftLiving','View'},'QueryPoints',[pt1 pt2]);
%view(140,30) % Modify the viewing angle
figure
pt3 = linspace(min(X2(:,4)),max(X2(:,4)),50)';
pt4 = linspace(min(X2(:,6)),max(X2(:,6)),50)';
ax2 = plotPartialDependence(Mdl2,{'SqftLot','Life'},'QueryPoints',[pt3 pt4]);
%view(140,30)