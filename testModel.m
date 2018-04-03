function [meanAcc, stdAcc] = testModel(testdata, w)
chunk = 1000; % chunk the testing data into smaller size and proceed the testing sequentially to prevent Matlab Out-of-Memory


N = size(testdata,1);
num = size(w,2); % the number of models

pred_y = zeros(N,num);
for i = 1:ceil(N/chunk)
    ind_start = 1+(i-1)*chunk;
    if i*chunk<N
        ind_end = i*chunk;
    else
        ind_end = N;
    end
    
    pred_v = testdata(ind_start:ind_end,1:end-1)*w;
    pred_v(pred_v > 0) = 1;
    pred_v(pred_v <= 0) = -1;

    pred_y(ind_start:ind_end,:) = pred_v;
end

err =  pred_y - testdata(:,end)*ones(1,num);
acc = sum(err==0,1)/N;
meanAcc = mean(acc)*100;
stdAcc = std(acc)*100;


