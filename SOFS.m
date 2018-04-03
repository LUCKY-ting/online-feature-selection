clc
clear
load('datasets/arcene.mat');
load('datasets/arceneTest.mat');
[n,d] = size(data);
epoch = ceil(2*d/n);
times = 10;  % run 10 times for calculating mean accuracy

% parameters
gamma = 1e+05;
B_range = 100:100:1000;
nn1 = size(B_range, 2);

acc_all = zeros(nn1,1);
record = zeros(nn1,11);

parfor q = 1:nn1
    
    B = B_range(q);
    errorNum = zeros(times,1);
    feaNum = zeros(times,epoch); % feaNum
    meanLoss = zeros(times,epoch); % mean loss
    allw = zeros(d-1, times);
    
    sr = RandStream.create('mt19937ar','Seed',1);
    RandStream.setGlobalStream(sr);
    
    startime = cputime;
    
    for run = 1:times
        errNum = 0;
        mu = zeros((d-1),1);
        sigma = ones((d-1),1);
        loss = 0;
        for o = 1:epoch
            index = randperm(n);
            for i=1:n
                j = index(i);
                x = data(j,1:d-1)';
                y = data(j,d);
                t = (o - 1)*n + i;
                
                pred_v = mu'* x;
                if pred_v > 0  %online prediction
                    pred_y = 1;
                else
                    pred_y = -1;
                end
                
                if y~=pred_y  %calculate error
                    errNum = errNum + 1;
                end
                
                loss = loss + (max(0, 1 - y*pred_v))^2;
                if y*pred_v < 1
                    belta = 1 / ((x.^2)'*sigma + gamma);
                    alpha = max(0, 1 - y*pred_v)*belta;
                    mu = mu + alpha * y * (sigma .* x);
                    sigma = 1./ ( 1./sigma + 1/gamma * (x.^2));
                    
                    % truncation operation
                    if size(sigma,1) > B
                        [val_sort,id] = sort(sigma,'descend');
                        mu(id(1: size(val_sort,1) - B )) = 0;
                    end
                    
                end
            end
            
            meanLoss(run,o) = loss/(o*n);
            feaNum(run,o) = nnz(mu);
        end
        
        errorNum(run) = errNum;
        allw(:,run) = mu;
    end
    
    duration = cputime - startime;
    
    accRate = (1 - errorNum./(n*epoch))*100;
    accMean = mean(accRate);
    accStd = std(accRate);
    
    %-------------test model performance on test data-------------------------
    [testAcc, testStd]= testModel(testData, allw);
    
    %-------------save result---------------------------------------
    acc_all(q) = accMean;
    record(q,:) = [gamma, B, duration/(times), mean(meanLoss(:,end)), std(meanLoss(:,end)), round(mean(feaNum(:,end))), round(std(feaNum(:,end))), accMean,accStd, testAcc, testStd];
end


fid = fopen('arcene_SOFS.txt','a');
fprintf(fid,'name = arcene, SOFS,  epoch= %d, runTimes= %d\n', epoch, times);
fprintf(fid,'gamma, B, duration[s], final loss + std,  final feaNum+std,  acc+std, testAcc+testStd \n');
for id =1: nn1
    fprintf(fid,'%g, %d, %.2f, %.4f, %.4f, ', record(id,1), record(id,2), record(id,3), record(id,4), record(id,5));
    fprintf(fid,' %d, %d, %.3f, %.3f, %.2f, %.2f \n', record(id,6), record(id,7), record(id,8), record(id,9), record(id,10), record(id,11));
end
fclose(fid);


