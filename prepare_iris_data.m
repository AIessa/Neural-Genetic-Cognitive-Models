%{
DATA PREPARATION

Instructions
STEP1: Import files to matlab: train.txt & test.txt (with no modifications, simply as tables)
       Either manually or running the "importtrain.m" and "importtest.m"
       scripts WITH THE APPROPRIATE path.
STEP2: Run this script.
STEP3: Open & run Perceptron script. 

This script preprocesses the training & test set of the Iris data:
- Both are turned from tables into matrices
- Training data is shuffled (rows)
- values in "desired output" vectors are transformed from categorical to
  integer/double using the numCat function (defined at end of script)
  Code of iris species is (in respective columns):
        -Iris setosa: 1,0,0
        -Iris versicolor: 0,1,0
        -Iris virginica: 0,0,1
%}


%TRAINING DATA:

train_matrix = table2array(train(:,1:4)); %separate integer part to trainingmatrix
train_numCategories = transformer(table2array(train(:,5))); %turn categorical label column into three integer columns
%randomize order of rows in training matrix:
train_merged = [train_matrix train_numCategories]; %re-merge all integer data
train_shuffled = train_merged(randperm(size(train_merged, 1)), :); %shuffle data

%TEST DATA:
test_matrix = table2array(test(:,1:4));
test_numCategories = transformer(table2array(test(:,5)));


%%
%FUNCTION DEFINITION:

%transforms categories to integer values:
function numCat = transformer(vector)
    index = 1;
    l = length(vector);
    numCat = [];
    while index<=l
        if vector(index)=="Iris-setosa"
            numCat = [numCat; 1, 0, 0];
        elseif vector(index)=="Iris-versicolor"
            numCat = [numCat; 0, 1, 0];
        else
            numCat = [numCat; 0, 0, 1]; %iris virginica
        end
        index=index+1;
    end
end
