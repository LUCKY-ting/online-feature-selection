clc
clear
load('datasets/dexter.mat');
load('datasets/dexterTest.mat');
[n,d] = size(data);
epoch = ceil(2*d/n); 
times = 10;  % run 10 times for calculating mean accuracy

% parameters
delta = 10.^(-2);
lambda = 10.^(-2);
eta  = 10.^(-4);
B_range = 100:100:1000;

nn2 = size(B_range, 2);

loss_all = zeros(nn2,1);
record = zeros(nn2,13);

parfor q = 1:nn2
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
        w = zeros((d-1),1);
        loss = 0;
        s_t = zeros((d-1),1);
        H_t = zeros((d-1),1);
        sum_g_t = zeros((d-1),1);
        for o = 1:epoch
            index = randperm(n);
            for i=1:n
                j = index(i);
                x = data(j,1:d-1)';
                y = data(j,d);
                t = (o - 1)*n + i;
                
                pred_v = w' * x;
                loss = loss + (max(0, 1 - y * pred_v))^2 + lambda/2 * sum(w.^2);
                if pred_v > 0  %online prediction
                    pred_y = 1;
                else
                    pred_y = -1;
                end
                
                if y~=pred_y  %calculate error
                    errNum = errNum + 1;
                end
                
                if y*pred_v < 1
                    g_t = - y * x * 2 *(1 - y*pred_v);
                    sum_g_t = sum_g_t + g_t;
                    s_t = sqrt( s_t.^2 + g_t.^2);
                    H_t = delta + s_t;
                else
                    g_t = zeros(d-1,1);
                end
                
                w = -eta * ( 1./ (lambda*eta*t + H_t) .* sum_g_t);
                
                % truncation operation
                if nnz(w) > B
                    [rows, ~, val_w] = find((w.^2).*H_t);
                    [val_sort,id] = sort(val_w,'ascend');
                    w(rows(id(1: size(val_sort,1) - B ))) = 0;
                end
            end
            
            meanLoss(run,o) = loss/(o*n);
            feaNum(run,o) = nnz(w);
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
    
     %-------------save results-------------------------
    loss_all(q) = mean(meanLoss(:,end)) + std(meanLoss(:,end));
    record(q,:) = [delta,lambda, eta, B, duration/(times), mean(meanLoss(:,end)), std(meanLoss(:,end)),round(mean(feaNum(:,end))), round(std(feaNum(:,end))), accMean,accStd, testAcc, testStd];
    
end
%-------------output result to file----------------------------------------
fid = fopen('dexter_B_ARDA.txt','a');
fprintf(fid,'name = dexter, B_ARDA,  epoch= %d, runTimes= %d\n', epoch, times);
fprintf(fid,'delta, lambda, eta, B, duration[s], final loss +std,  final feaNum+std,  acc+std, testAcc+testStd \n');
for ll =1 : nn2
    fprintf(fid,'%g, %g, %g, %d, %.2f, %.4f, %.4f, ', record(ll,1), record(ll,2), record(ll,3), record(ll,4), record(ll,5), record(ll,6), record(ll,7));
    fprintf(fid,' %d, %d, %.3f, %.3f, %.2f, %.2f \n', record(ll,8), record(ll,9), record(ll,10), record(ll,11), record(ll,12),record(ll,13));
end
fclose(fid);


