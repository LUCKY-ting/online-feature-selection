clc
clear
load('datasets/dexter.mat');
load('datasets/dexterTest.mat');
[n,d] = size(data);
epoch = ceil(2*d/n);
times = 10;  % run 10 times for calculating mean accuracy

% parameters
lambda = 1e-2; % regularization parameter
eta = 10^(-6.5);
B_range = 100:100:1000; % the number of features kept in feature reduction

nn1 = size(B_range,2);
loss_all = zeros(nn1,1);
record = zeros(nn1,12);

parfor p = 1: nn1
    B = B_range(p);
    sr = RandStream('mt19937ar','Seed',1);
    RandStream.setGlobalStream(sr);
    
    errorNum = zeros(times,1);
    feaNum = zeros(times,epoch);
    allw = zeros(d-1, times);
    avgCumLoss =  zeros(times,epoch);
    startime = cputime;
    
    for run = 1:times
        errNum = 0;
        w = zeros((d-1),1);
        loss = 0;
        for o = 1:epoch
            index = randperm(n);
            for i=1:n
                j = index(i);
                x = data(j,1:d-1)';
                y = data(j,d);
                t = (o - 1)*n + i;
                
                pred_v = w' * x;
                loss = loss + max(0, 1 - y * pred_v) + lambda/2*sum(w.^2);
                if pred_v > 0  %online prediction
                    pred_y = 1;
                else
                    pred_y = -1;
                end
                
                if y~=pred_y  %calculate error
                    errNum = errNum + 1;
                end
                
                if y*pred_v < 1
                    w = (1 - lambda*eta)*w + eta*y*x;
                    l2norm = norm(w,2);
                    w = min(1, 1/(sqrt(lambda)*l2norm)) * w;
                    nnzNum = nnz(w);
                    if nnzNum > B
                        [rows,~, val_w] = find(abs(w));
                        [sort_w,id] = sort(val_w,'ascend');
                        w(rows(id(1:nnzNum-B))) = 0;
                    end
                else
                    w = (1 - lambda*eta)*w;
                end
            end
            
            feaNum(run,o) =  nnz(w);
            avgCumLoss(run,o) = loss/(o*n);
        end
        
        errorNum(run) = errNum;
        allw(:,run) = w;
    end
    
    duration = cputime - startime;
    
    accRate = (1 - errorNum./(n*epoch))*100;
    accMean = mean(accRate);
    accStd = std(accRate);
    
    %-------------test model performance on test data-------------------------
    [testAcc, testStd]= testModel(testData, allw);
    
    %-----------store results--------------------------
    loss_all(p) = mean(avgCumLoss(:,end));
    record(p,:) = [lambda, eta, B,duration/(times), mean(avgCumLoss(:,end)), std(avgCumLoss(:,end)) ,round(mean(feaNum(:,end))), round(std(feaNum(:,end))), accMean,accStd, testAcc, testStd];
end

fid = fopen('dexter_OFS.txt','a');
fprintf(fid,'name = dexter, OFS, epoch= %d, runTimes= %d\n', epoch, times);
fprintf(fid,'lambda, eta, B, duration[s], final loss +std,  final feaNum+std,  acc+std, testAcc+testStd \n');
for ll = 1: nn1
    fprintf(fid,'%g, %g, %d, %.2f, %.4f, %.4f, ', record(ll,1), record(ll,2), record(ll,3),record(ll,4), record(ll,5), record(ll,6));
    fprintf(fid,' %d, %d, %.3f, %.3f, %.2f, %.2f \n', record(ll,7), record(ll,8), record(ll,9), record(ll,10), record(ll,11), record(ll,12));
end
fclose(fid);


