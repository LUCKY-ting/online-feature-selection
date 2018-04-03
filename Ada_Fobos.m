clc
clear
load('datasets/dexter.mat');
load('datasets/dexterTest.mat');
[n,d] = size(data);
epoch = ceil(2*d/n);
times = 10;  % run 10 times for calculating mean accuracy

% parameters
delta = 10.^(-2);
eta = 10.^(-4);
lambda_range = [0.1:0.1:1];
nn1 = size(lambda_range,2); 

parfor p = 1 : nn1
    lambda = lambda_range(p);
    
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
        for o = 1:epoch
            index = randperm(n);
            for i=1:n
                j = index(i);
                x = data(j,1:d-1)';
                y = data(j,d);
                t = (o - 1)*n + i;
                
                pred_v = w' * x;
                loss = loss + max(0, 1 - y * pred_v)^2 + lambda * norm(w,1);
                if pred_v > 0  %online prediction
                    pred_y = 1;
                else
                    pred_y = -1;
                end
                
                if y~=pred_y  %calculate error
                    errNum = errNum + 1;
                end
                
                if y*pred_v < 1
                    g_t = - 2*y*x*(1 - y*pred_v);
                    s_t = sqrt(s_t.^2 + g_t.^2);
                    H_t = delta + s_t;
                    iw = w - eta./ H_t .* g_t;
                else
                    iw = w;
                end
                
                w = sign(iw) .* max(0, abs(iw) - lambda*eta ./ H_t);
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
    
    %-------------output result to file----------------------------------------
    fid = fopen('dexter_Ada_Fobos.txt','a');
    fprintf(fid,'name = dexter, Ada_Fobos,  epoch= %d, runTimes= %d\n', epoch, times);
    fprintf(fid,'delta, lambda, eta, duration[s], meanLoss+std, final feaNum+std, acc+std, testAcc+testStd \n');
    fprintf(fid,'%g, %g, %g, %.2f, %.3f, %.3f,', delta, lambda, eta, duration/(times), mean(meanLoss(:,end)), std(meanLoss(:,end)));
    fprintf(fid,' %d, %d, %.4f, %.4f, %.2f, %.2f \n', round(mean(feaNum(:,end))), round(std(feaNum(:,end))), accMean, accStd, testAcc, testStd);
    fclose(fid);

end

