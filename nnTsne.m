function [ rgbData, net, tsneReduced, reducedNeuralFull, autoencoder ] = nnTsne( data, distance )
%function to perfrom neural network t-SNE as described in 10.1021/acs.analchem.5c01812

%inputs; - data = the data matrix of samples * variables to reduce
%           distance = the distance matrix to use when performing the initial t-SNE


%outputs; - rgbData = and N*3 matrix of the the full reduction scaled to 0-1 in each dimension
%           net = the neural network 
%           tSneReduced = the initial t-SNE used to train the neural network
%           reducedNeuralFull = the full dataset reduced to the 3 dimensional space (not scaled 0-1)
%           autoencoder = the trained autoencoders


%% Perform the initial subsampling and reduction

%get a subsample of ~5000 sample as every 1/5000th sample
subsetSize = ceil(size(data,1) ./ 5000);

%get the subet of data
subset = data(1:subsetSize:size(data,1),:);

%reduce the subset with PCA
[Coeffs, Scores] = pca(subset);
%get the top 50 scores
if size(Scores,2) > 50
    top50Scores = Scores(:,1:50);
    [~, mu, ~] = zscore(subset);
    top50Coeffs = Coeffs(:,1:50);
else
    top50Scores = Scores(:,1:size(Scores,2));
    [~, mu, ~] = zscore(subset);
    top50Coeffs = Coeffs(:,1:size(Scores,2));
end

%do t-SNE on the subset
tsneReduced = tsne(subset, 'distance', distance, 'NumDimensions',3);

%reduce the full data into the top 50 components space of the subset
pcaReduced50Full = (data - repmat(mu,size(data,1),1)) * top50Coeffs;

%train an autoencoder on the top 50 scores to 25 dimensions
autoencoder{1} = trainAutoencoder(top50Scores',25, ...
    'MaxEpochs',1000, ...
    'EncoderTransferFunction', 'logsig',...
    'DecoderTransferFunction','logsig', ...
    'L2WeightRegularization', 0.0001, ...
    'SparsityRegularization', 8, ...
    'UseGPU',true, ...
    'SparsityProportion',0.1);
%encode the subset
featureSpaceSubset{1} = encode(autoencoder{1},top50Scores');
%encode the full data
featureSpaceFull{1} = encode(autoencoder{1}, pcaReduced50Full');
%train an autoencoder on the 25 dimensions down to a further 10
autoencoder{2} = trainAutoencoder(featureSpaceSubset{1},10, ...
    'MaxEpochs',1000, ...
    'EncoderTransferFunction', 'logsig',...
    'DecoderTransferFunction','logsig', ...
    'L2WeightRegularization', 0.01, ...
    'SparsityRegularization', 4, ...
    'UseGPU',true, ...
    'SparsityProportion',0.1);
%encode the subset
featureSpaceSubset{2} = encode(autoencoder{2},featureSpaceSubset{1});
%encode the full data
featureSpaceFull{2} = encode(autoencoder{2},featureSpaceFull{1});
%% set up the neural network

%set the Bayesian regularisation training method
trainingMethod = 'trainbr';
%initialise the network
neuralNetwork = fitnet(25, trainingMethod);
%set training and testing data sizes
neuralNetwork.divideParam.trainRatio = 70/100;
neuralNetwork.divideParam.valRatio = 15/100;
neuralNetwork.divideParam.testRatio = 15/100;
%train the network
[net, ~] = train(neuralNetwork,featureSpaceSubset{2},tsneReduced');
%embed the subset
subsetNeural = sim(net,featureSpaceSubset{2});
%embed the full data
reducedNeuralFull = sim(net,featureSpaceFull{2});
%scale to 0-1 of the original t-SNE
for k = 1:3
    temp = tsneReduced(:,k);
    a = min(temp);
    b = max(temp - min(temp));
    temp = reducedNeuralFull(k,:);
    temp2 = (temp - a) ./ b;
    rgbData(:,k) = temp2;
end


end