%%PERCEPTRON
%{
TRAINING & TESTING PERCEPTRON (simple feedback)

Please run "prepare_iris_data.m" before running this script, as it includes
instructions to import the data & preprocesses it to be fed into the
perceptron.
%}

%create empty log file:
logfile = fopen('Perceptron_log.txt','w');
fprintf(logfile,'Log file for Perceptron performance on iris data \n');

%%
%STEP1: SET PARAMETERS
%{
The function trainpercept is defined to take the following input:
- inMat = train_shuffled(:,1:4)
- outVec = train_shuffled(:,5:7)
- bias = -1; (this is arbitrary, there is an associated weight)
- learningrate = 0.5 (weight adjustment after each iteration)
- w0 = -1*2.*rand(5,1); (initial, randomly generated weight vector)
- iterations/epochs = (stop criterion for this perceptron)
%}

bias=-1;
l=0.7;
w0=-1*2.*rand(5,1); %where first weight is initial theta in stepfunction
epochs=90;

fprintf(logfile,'Training perceptron with: \n');
fprintf(logfile,sprintf('bias=%g learningrate=%g weights=[%s,%s,%s,%s,%s] epochs=%g (stop criterion) \n',bias,l,w0,epochs));

%%
%STEP2:TRAINING
%{
Perceptron
    - 5 nodes in the inputlayer (for input vector containing petal width,
    petal length, sepal width, sepal lenght, bias)
    - 3 nodes in the outputlayer: Each has an activation function specific
    to one of the 3 types of iris -> it will activate for only one type of
    iris. 
    - Each node calculates: w1*x1+w2*x2+w3*x3+w4*x4+bias = out

(Note: Since it is possible to separate iris setosa from the other species 
only looking at petal width & length, we could reduce the input for this 
node. While this may make the perceptron more efficient, I didn't implement 
this detail.)

%}

myperceptron=trainperceptron(train_shuffled(:,1:4),train_shuffled(:,5:7),bias,l,w0,epochs);

%%
%STEP3:TESTING
%{
I use the same bias (-1) as before, but the new weight vector w
that resulted from the trained perceptron ("myperceptron"). This is
because I implemented it in a way that the bias has a weight itself,
hence I am actually using a modified weight value as well and not the
original one (as I am using the new weight vector).

%}

result=testperceptron(test_matrix,test_numCategories,-1,myperceptron,logfile);

%%
%STEP3: RESULTS

%to view as variable
result_table=array2table(result);

fprintf(logfile,sprintf('Classification results:\n'));
fprintf(logfile,'%6.0f %6.0f %6.0f\r\n',transpose(result));
fprintf(logfile,sprintf('Actual classes: \n'));
fprintf(logfile,'%6.0f %6.0f %6.0f\r\n',transpose(test_numCategories));

%Success rates:
%overall rate of (mis)classification:
%(only counts it as correct classification if is only correctly classified as ONE tpye of iris)
false_class=0;
true_class=0;
for row = 1:length(result)
    if result(row,1)==test_numCategories(row,1) && result(row,2)==test_numCategories(row,2) && result(row,3)==test_numCategories(row,3)
        true_class=true_class+1;
    elseif (result(row,1)~=test_numCategories(row,1)) || (result(row,2)~=test_numCategories(row,2)) || (result(row,3)~=test_numCategories(row,3))
        false_class=false_class+1;
    end
end
correct_percent=100*true_class/length(result);
fprintf(logfile,sprintf('The ANN correctly classified %g out of %g. Success rate of : %g percent. \n\n',true_class,length(result),correct_percent));

%confusion matrix for each class
    cMset = score(result,test_numCategories,1);
    cMvers = score(result,test_numCategories,2);
    cMvirg = score(result,test_numCategories,3);
    fprintf(logfile,sprintf('confusion matrix for classification of iris setosa: FP: %g, FN: %g, TP: %g, TN: %g \n',cMset(1),cMset(2),cMset(3),cMset(4)));
    fprintf(logfile,sprintf('confusion matrix for classification of iris versicolor: FP: %g, FN: %g, TP: %g, TN: %g \n',cMvers(1),cMvers(2),cMvers(3),cMvers(4)));
    fprintf(logfile,sprintf('confusion matrix for classification of iris virginica: FP: %g, FN: %g, TP: %g, TN: %g \n',cMvirg(1),cMvirg(2),cMvirg(3),cMvirg(4)));

    

%%
%FUNCTIONS

%TRAINING PERCEPTRON:
%   -inMat -> input matrix without desired output/class
%   -outMat -> the last 3 columns of the input matrix with class codes
function train = trainperceptron(inMat,outMat,bias,learningrate,w,epochs)
    
    numIn=length(inMat); %for the second for loop
    %instantiating weights for each (output) node
    w1=w;
    w2=w;
    w3=w;
    
    for i= 1:epochs
        
        for row = 1:numIn 
        %loops through matrix row-wise
        %all nodes see all rows
        %weights for each output node are updated row per row
            
            %OUTPUTNODE1:   
            %only cares about first (index=5) output column, 1 for iris
            %setosa, 0 for other types
            w1=node(bias,w1,inMat(row,:),outMat(row,:),1,learningrate); 
                %5 stands for the index for outMat. Technically, I could 
                %just write outMat(5), but I found it easier to handle this way.
            
            %OUTPUTNODE2:   
            %only cares about second (index=6) output column, 1 for iris
            %versicolor, 0 for other types
            w2=node(bias,w2,inMat(row,:),outMat(row,:),2,learningrate); 
            
            %OUTPUTNODE3:   
            %only cares about third (index=7) output column, 1 for iris
            %virginica, 0 for other types
            w3=node(bias,w3,inMat(row,:),outMat(row,:),3,learningrate); 

        end
       
    end
    
    train = [w1 w2 w3]; %output matrix with trained weights for each (output) node
end

%A NODE (+SIMPLE FEEDBACK LEARNING):
%   takes one row of input. The out_vec has 3 columns & the index is node
%   specific, such that output node 1 looks at column 5 of out_vec, node 2
%   at column 6 and node 3 at column 7.
function new_weights = node(bias,w,in_row,out_vec,index,learningrate)
    
    activation=bias*w(1,1)+in_row(1)*w(2,1)+in_row(2)*w(3,1)+in_row(3)*w(4,1)+in_row(4)*w(5,1);
    
    %classification with step function:
    theta=w(1,1); %threshold (changes when weight vector changes)
    if activation >= theta
        y=1;
    else
        y=0;
    end
            
            
    %simple feedback learning (weight correction):
    if y==1 && out_vec(index)==0 %e.g. classified as setosa but shouldn't be
        w(1,1)=w(1,1) - learningrate*bias;
        w(2,1)=w(2,1) - learningrate*in_row(1);
        w(3,1)=w(3,1) - learningrate*in_row(2);
        w(4,1)=w(4,1) - learningrate*in_row(3);
        w(5,1)=w(5,1) - learningrate*in_row(4);
    elseif y==0 && out_vec(index)==1 %e.g. not classified as setosa but should be
        w(1,1)=w(1,1) + learningrate*bias;
        w(2,1)=w(2,1) + learningrate*in_row(1);
        w(3,1)=w(3,1) + learningrate*in_row(2);
        w(4,1)=w(4,1) + learningrate*in_row(3);
        w(5,1)=w(5,1) + learningrate*in_row(4);
    end
    
    new_weights=w; %output updated weight vector
end

%TESTING PERCEPTRON:
function test = testperceptron(testinMat,testoutMat,bias,trained_wMat,logfile)
    
    numIn=length(testinMat); %for the second for loop
    %instantiating weights for each (output) node
    w1=trained_wMat(:,1);
    w2=trained_wMat(:,2);
    w3=trained_wMat(:,3);
    
    outputMatrix=[];
    
    for row = 1:numIn 
    %loops through matrix row-wise
    %all nodes see all rows
    %weights for each output node are updated row per row

        %OUTPUT1:   
        %should only return 1 for iris setosa, 0 for other types
        output1=testnode(bias,w1,testinMat(row,:),testoutMat(row,:),1,row,1,logfile);

        %OUTPUTNODE2:   
        %should only return 1 for iris versicolor, 0 for other types
        output2=testnode(bias,w2,testinMat(row,:),testoutMat(row,:),2,row,2,logfile); 

        %OUTPUTNODE3: 
        %should only return 1 for iris virginica, 0 for other types
        output3=testnode(bias,w3,testinMat(row,:),testoutMat(row,:),3,row,3,logfile);
        
        row_output=[output1 output2 output3];
        %fprintf(logfile,sprintf('row number %g : Perceptron output is [%s,%s,%s]. Actual class is [%s,%s,%s]. \n',row,row_output,testoutMat(row,:)));
        outputMatrix=[outputMatrix;row_output];
    end
    test = outputMatrix; %output matrix with "output code" that the perceptron calculated for each row
end


%A TEST-NODE (OUTPUT is CLASSIFICATION, NO LEARNING):
%   takes one row of input. The out_vec has 3 columns & the index is node
%   specific, such that output node 1 looks at column 5 of out_vec, node 2
%   at column 6 and node 3 at column 7.
function classify = testnode(bias,w,in_row,out_vec,index,rowNumber,nodeNumber,logfile)
    
    activation=bias*w(1,1)+in_row(1)*w(2,1)+in_row(2)*w(3,1)+in_row(3)*w(4,1)+in_row(4)*w(5,1);
    
    %classification with step function:
    theta=w(1,1); %threshold (changes when weight vector changes)
    if activation >= theta
        y=1;
    else
        y=0;
    end
    %sprintf('node: %g, row: %g, actual class: %g, assigned class: %g \n',nodeNumber, rowNumber, out_vec(index),y)
    classify=y; %output classification
end

%CREATES CONFUSION MATRIX:
function confusionmatrix = score(output_result,true_result,type)
        false_pos=0;
        false_neg=0;
        true_pos=0;
        true_neg=0;
        for row = 1:length(output_result)
            if output_result(row,type) == 1 && true_result(row,type)==1
                true_pos=true_pos+1;
            elseif output_result(row,type) == 1 && true_result(row,type)==0
                false_pos=false_pos+1;
            elseif output_result(row,type) == 0 && true_result(row,type)==1
                false_neg=false_neg+1;
            elseif output_result(row,type) == 0 && true_result(row,type)==0
                true_neg=true_neg+1;
            end
        end
        confusionmatrix=[false_pos, false_neg, true_pos, true_neg];
end



