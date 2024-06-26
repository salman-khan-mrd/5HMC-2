clc;
close all;
clear all;

%+++++++++++++++++ File Loading +++++++++++
load prediction;
classif_tech=yy;

%+++++++++++++++++ Variable declaration ++++++++++
TP=0;TN=0;FN=0;FP=0;f=0;c1=0;c2=0;

%+++++++++++++++ Class Label++++++++++++++
 
 DNA_labels(1:478)=1;
  DNA_labels(479:1050)=2;
 
%++++++++++++++++++++++++++++++++++++++++++
 
y=classif_tech;
Result=find(y==DNA_labels);
Total_correct=size(Result,2);
Accuracy=(Total_correct/1050)*100;


   %+++++++++ individual Accuracy ++++++++++++++
    for i=1:478
        if( y(i)==1)
            c1=c1+1;
        end
    end 
    for i=479:1050
        if(y(i)==2)
            c2=c2+1;
        end 
    end

    
    C1=(c1/1418)*100
    C2=(c2/1418)*100
    
%+++++++++++++++++ Performance Measure Calculation+++++++++++++++++++++
for i=1:size(DNA_labels,2)
    if DNA_labels(1,i)==1 && y(1,i)==1
        TP=TP+1;
    elseif DNA_labels(1,i)==2 && y(1,i)==2
        TN=TN+1;
    elseif DNA_labels(1,i)==1 && y(1,i)==2
        FN=FN+1;
    elseif DNA_labels(1,i)==2 && y(1,i)==1
        FP=FP+1;
    end
end
%++++++++++++++ Accuracy +++++++++++++++
Acc=(TP+TN)/(TP+TN+FN+FP)*100

%++++++++++++++ Sensitivity ++++++++++++
sen=TP/(TP+FN)*100

%++++++++++++++ Specificity +++++++++++++
spe=TN/(TN+FP)*100

%+++++++++++++++ Balance Accuracy+++++++++
BAcc= (sen+spe)/2
%+++++++++++++++ MCC ++++++++++++++++++
    a=(TP+FN)*(TP+FP)*(TN+FN)*(TN+FP);
MCC=((TP*TN)-(FN*FP))/sqrt(a)

%++++++++++++++++++ F-measrue++++++++++++
    gh=(TP/(TP+FP)); %++++++++++ This is Precision
    rh=(TP/(TP+FN));  % ++++++++ This is Recall
    F=(gh*rh)/(gh+rh);
    F_Measure=2*F
 
%+++++++++++++++ Precision & Recall +++++++++++
    Precision=gh
    Recall=rh
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++