%Assignment 7-9 - Means 

%% Get Data
table = readtable('train.csv','Delimiter',',');
train = table{:,2:94}; %table2array

%% Class Ranges (needed because of how data was imported)
C1 = 1:1929;
C2 = 1930:18051;
C3 = 18052:26055;
C4 = 26056:28746;
C5 = 28747:31485;
C6 = 31486:45620;
C7 = 45621:48459;
C8 = 48460:56923;
C9 = 56924:61878;

%Label Vector
label = zeros(61878,1);
label(C1,1) = 1; %Class 1
label(C2,1) = 2; %Class 2
label(C3,1) = 3; %Class 3
label(C4,1) = 4; %Class 4
label(C5,1) = 5; %Class 5
label(C6,1) = 6; %Class 6
label(C7,1) = 7; %Class 7
label(C8,1) = 8; %Class 8
label(C9,1) = 9; %Class 9
%%
figure(1) %Class Distributions
hist(label,[1 2 3 4 5 6 7 8 9])
xlabel('Classes')
ylabel('Data Points')
title('Class-Data Points Distribution')
grid

%%
%Count of non zero elements for each feature per class
class(1,:) = sum( train(C1,:) ~= 0, 1);
class(2,:) = sum( train(C2,:) ~= 0, 1);
class(3,:) = sum( train(C3,:) ~= 0, 1);
class(4,:) = sum( train(C4,:) ~= 0, 1);
class(5,:) = sum( train(C5,:) ~= 0, 1);
class(6,:) = sum( train(C6,:) ~= 0, 1);
class(7,:) = sum( train(C7,:) ~= 0, 1);
class(8,:) = sum( train(C8,:) ~= 0, 1);
class(9,:) = sum( train(C9,:) ~= 0, 1);

figure(2) %Classes 1-3
for i = 1:3
subplot(3,1,i)
bar(class(i,:))
ylabel({'Class' num2str(i)})
if i == 3
    xlabel('Features')  
end
title('Feature data per Class')
end

figure(3) %Classes 4-6
for i = 4:6
subplot(3,1,i-3)
bar(class(i,:))
ylabel({'Class' num2str(i)})
if i == 6
    xlabel('Features')  
end
title('Feature data per Class')
end

figure(4) %Classes 7-9
for i = 7:9
subplot(3,1,i-6)
bar(class(i,:))
ylabel({'Class' num2str(i)})
if i == 9
    xlabel('Features')  
end
end
%%
figure(5) %Class vs Features Stacked
bar(1:93,class','stacked')
legend('Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9')
xlabel('Features')
title('Feature data per class')
grid
%%
figure(6) %Feature vs Class 1 - 32
for i = 1:32
subplot(4,8,i)
h = bar(diag(class(:,i)'),'stacked');
title({num2str(i)});
set(gca, 'YTick', [-inf inf]); %get rid of y-axis
end

figure(7) %Feature vs Class 33 - 64
for i = 33:64
subplot(4,8,i-32)
h = bar(diag(class(:,i)'),'stacked');
title({num2str(i)});
set(gca, 'YTick', [-inf inf]); %get rid of y-axis
end

figure(8) %Feature vs Class 65 - 93
for i = 65:93
subplot(4,8,i-64)
h = bar(diag(class(:,i)'),'stacked');
title({num2str(i)});
set(gca, 'YTick', [-inf inf]); %get rid of y-axis
end