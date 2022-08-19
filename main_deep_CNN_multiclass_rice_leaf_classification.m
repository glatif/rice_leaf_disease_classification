clear all
close all
clc



ExeType = 2;

if (ExeType == 1) % Non-Freezed Normalized
    % GPU device selection
    g = gpuDevice(1);
    reset(g);
    SaveResults='Output_Results_NonFreezed_Normalized_Enhanced';
    result = isfolder(SaveResults);
    if result==0
        mkdir Output_Results_NonFreezed_Normalized_Enhanced;
    end
    diary 'deep_CNN_Multi_All_methods_results_NonFreeze_Normalized_Enhanced.txt'
elseif (ExeType == 2) % Non-Freezed Non Normalized
    % GPU device selection
    g = gpuDevice(4);
    reset(g);
    SaveResults='Output_Results_NonFreezed_Normalized';
    result = isfolder(SaveResults);
    if result==0
        mkdir Output_Results_NonFreezed_Normalized;
    end
    diary 'deep_CNN_Multi_All_methods_results_NonFreezed_Normalized_Google.txt'
elseif (ExeType == 3) % Non-Freezed Non Normalized Augmented
    % GPU device selection
    g = gpuDevice(3);
    reset(g);
    SaveResults='Output_Results_NonFreezed_NonNormalized_Augment';
    result = isfolder(SaveResults);
    if result==0
        mkdir Output_Results_NonFreezed_NonNormalized_Augment;
    end
    diary 'deep_CNN_Multi_All_methods_results_NonFreeze_NonNormalized_Augmented.txt'
elseif (ExeType == 4) % Freezed Non Normalized
    % GPU device selection
    g = gpuDevice(4);
    reset(g);
    SaveResults='Output_Results_Freezed_NonNormalized';
    result = isfolder(SaveResults);
    if result==0
        mkdir Output_Results_Freezed_NonNormalized;
    end
    diary 'deep_CNN_Multi_All_methods_results_Freezed_NonNormalized.txt'
end
outputFolder = '';
% alexnet, googlenet, densenet201, resnet101, resnet50, vgg16, vgg19, resnet18,
% squeezenet,
%%%% openExample('nnet/TransferLearningUsingGoogLeNetExample')


rootFolder = fullfile(outputFolder, 'LabelledRice');
categories = {'BrownSpot', 'Healthy', 'Hispa', 'LeafBlast'};
numClasses=4;
fileSaveName = 'CNN_Train_Progress.tif';
methodName={'googlenet','vgg16', 'vgg19', 'densenet201', 'alexnet'}; % 'vgg16', 'vgg19', 'densenet201', 'alexnet', 'googlenet', 'Method_1', 'lenet'
epochesSet=[30]; % 8
batchsizeSet=[200]; %100, 500

SubDataSet={'LabelledRice'}; % 'T1', 'T1c', 'T2', 'Flair'
AugmEnable =0;

myTrainingFolder = 'C:\Research\Rice_Disease\5classDataset\RiceLeafsv3\train';



for iq=1:size(methodName,2)
    methodName(1,iq)
%     if strcmp(methodName(1,iq), 'vgg16')
%         net = vgg16();
%     elseif strcmp(methodName(1,iq), 'alexnet')
%         net = alexnet();
%     elseif strcmp(methodName(1,iq), 'googlenet')
%         net = googlenet();
    for jq=1:size(epochesSet,2)
        mepo=epochesSet(1,jq);
        for kq=1:size(batchsizeSet, 2)
            MinBatchS=batchsizeSet(1,kq);
            if strcmp(methodName(1,iq), 'vgg16')
                MinBatchS = MinBatchS/5;
            end
            if strcmp(methodName(1,iq), 'vgg19')
                MinBatchS = MinBatchS/5;
            end
            for lq=1:size(SubDataSet, 2)
                rootFolder = fullfile(outputFolder, SubDataSet(1,lq))
                fileSaveName=strcat(SaveResults,'\',methodName(1,iq), '_', int2str(mepo), '_', int2str(MinBatchS), '_', SubDataSet(1,lq), '.tif');

                "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
                
                %imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
                imds = imageDatastore(myTrainingFolder,'IncludeSubfolders', true,  'LabelSource', 'foldernames');
                tbl = countEachLabel(imds)
                if (ExeType == 2)
                    minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category
                    % Use splitEachLabel method to trim the set.
                    imds = splitEachLabel(imds, minSetCount, 'randomize');
                end
                % Notice that each set now has exactly the same number of images.
                countEachLabel(imds)
                if strcmp(methodName(1,iq), 'vgg16')
                    imds.ReadFcn = @(filename)readAndPreprocessImageVGG16(filename);
                    imageSize = [224 224 3];
                elseif strcmp(methodName(1,iq), 'vgg19')
                    imds.ReadFcn = @(filename)readAndPreprocessImageVGG16(filename);
                    imageSize = [224 224 3];
                elseif strcmp(methodName(1,iq), 'alexnet')
                    imds.ReadFcn = @(filename)readAndPreprocessImageAlexNet(filename);
                    imageSize = [227 227 3];
                elseif strcmp(methodName(1,iq), 'densenet201')
                    imds.ReadFcn = @(filename)readAndPreprocessImageDenseNet201(filename);
                    imageSize = [224 224 3];
                elseif strcmp(methodName(1,iq), 'googlenet')
                    imds.ReadFcn = @(filename)readAndPreprocessImageGoogleNet(filename);
                    imageSize = [224 224 3];
                elseif strcmp(methodName(1,iq), 'lenet')
                    imds.ReadFcn = @(filename)readAndPreprocessImageLeNet(filename);
                    imageSize = [28 28 1];
                elseif strcmp(methodName(1,iq), 'Method_1')
                    imds.ReadFcn = @(filename)readAndPreprocessImageMethod_1(filename);
                    imageSize = [30 30 1];
                end

                [trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize'); %0.8
                [trainingSet, valSet] = splitEachLabel(trainingSet, 0.8, 'randomize'); %0.8
                %%%%%%%%%%% Data Augmentation
                if (ExeType == 3)
                    imageAugmenter = imageDataAugmenter('RandRotation',[-10,10], 'RandXTranslation',[-2 2], 'RandYTranslation',[-2 2])
                    %imageAugmenter = imageDataAugmenter('RandRotation',[-15,15],'RandXTranslation',[-4 4],'RandYTranslation',[-4 4], 'RandScale',[0.95 1])
                    trainingSet = augmentedImageDatastore(imageSize,trainingSet,'DataAugmentation',imageAugmenter);
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                %numTrainImages = numel(trainingSet.Labels)
                
                
                if strcmp(methodName(1,iq), 'vgg16')
                    net = vgg16();
                    inputSize = net.Layers(1).InputSize
                    layersTransfer = net.Layers(1:end-3);

                    NumFreezeLayers=size(net.Layers(1:end-3), 1);

                    %  weight updating - non -freezed
                    layersTransfer(1:10) = freezeWeights(layersTransfer(1:10));
                    % no weight update - freezed
                    %
                    if (ExeType == 4)
                        layersTransfer(1:NumFreezeLayers) = freezeWeights(layersTransfer(1:NumFreezeLayers));
                    end

                    % layersTransferAll = net;
                    %numClasses = numel(unique(categories(trainingSet.Labels)))
                    % layersnew = layerGraph(layersTransferAll);

                    lgraph1 = [
                        layersTransfer
                        fullyConnectedLayer(numClasses, 'WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                        softmaxLayer
                        classificationLayer];
                     lgraph1(39).Name='fc2';
                     lgraph1(40).Name='prob';
                     lgraph1(41).Name='ClassificationLayer_predictions';
                elseif strcmp(methodName(1,iq), 'vgg19')
                    net = vgg19();
                    inputSize = net.Layers(1).InputSize
                    layersTransfer = net.Layers(1:end-3);

                    NumFreezeLayers=size(net.Layers(1:end-3), 1);

                    %  weight updating - non -freezed
                    layersTransfer(1:10) = freezeWeights(layersTransfer(1:10));
                    % no weight update - freezed
                    %layersTransfer(1:NumFreezeLayers) = freezeWeights(layersTransfer(1:NumFreezeLayers));
                    if (ExeType == 4)
                        layersTransfer(1:NumFreezeLayers) = freezeWeights(layersTransfer(1:NumFreezeLayers));
                    end

                    % layersTransferAll = net;
                    %numClasses = numel(unique(categories(trainingSet.Labels)))
                    % layersnew = layerGraph(layersTransferAll);

                    lgraph1 = [
                        layersTransfer
                        fullyConnectedLayer(numClasses, 'WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                        softmaxLayer
                        classificationLayer];
                     lgraph1(45).Name='fc2';
                     lgraph1(46).Name='prob';
                     lgraph1(47).Name='ClassificationLayer_predictions';                
                elseif strcmp(methodName(1,iq), 'Method_1')
                    layers = [
                        imageInputLayer([30 30 1], 'Name','new_input')
                        convolution2dLayer(3,30,'Padding',0, 'Name','new_cnv1')
                        %batchNormalizationLayer('Name','bat_norm1')
                        %reluLayer('Name','new_relu1')
                        maxPooling2dLayer(2,'Stride',2, 'Name','new_maxp1')
                        convolution2dLayer(3,60,'Padding',0, 'Name','new_conv2')
                        dropoutLayer(.20, 'Name','drop_1')
                        convolution2dLayer(3,30,'Padding',0, 'Name','new_conv3')
                        maxPooling2dLayer(2,'Stride',2, 'Name','new_maxp2')
                        %batchNormalizationLayer('Name','bat_norm2')
                        %reluLayer('Name','new_relu2')
                        %maxPooling2dLayer(2,'Stride',2, 'Name','new_maxp2')
                        fullyConnectedLayer(750, 'Name','new_fc')
                        %flattenLayer('Name','flatten1')
                        fullyConnectedLayer(256, 'Name','new_fc2')
                        fullyConnectedLayer(64, 'Name','new_fc3')
                        fullyConnectedLayer(numClasses, 'Name','new_fc4')
                        softmaxLayer('Name','new_softmax')
                        classificationLayer('Name','new_classi_layer')];
                    
                    lgraph1 = layerGraph(layers);
                elseif strcmp(methodName(1,iq), 'lenet')
                    layers = [
                        imageInputLayer([28 28 1], 'Name','new_input')
                        convolution2dLayer(5,6,'Padding','same', 'Name','new_cnv1')
                        batchNormalizationLayer('Name','bat_norm1')
                        reluLayer('Name','new_relu1')
                        maxPooling2dLayer(2,'Stride',2, 'Name','new_maxp1')
                        convolution2dLayer(5,16,'Padding',0, 'Name','new_conv2')
                        batchNormalizationLayer('Name','bat_norm2')
                        reluLayer('Name','new_relu2')
                        maxPooling2dLayer(2,'Stride',2, 'Name','new_maxp2')
                        fullyConnectedLayer(400, 'Name','new_fc')
                        %flattenLayer('Name','flatten1')
                        fullyConnectedLayer(120, 'Name','new_fc2')
                        fullyConnectedLayer(84, 'Name','new_fc3')
                        fullyConnectedLayer(numClasses, 'Name','new_fc4')
                        softmaxLayer('Name','new_softmax')
                        classificationLayer('Name','new_classi_layer')];
                    
                    lgraph1 = layerGraph(layers);
                elseif strcmp(methodName(1,iq), 'densenet201')
                    net = densenet201();
                    if isa(net,'SeriesNetwork') 
                      lgraph = layerGraph(net.Layers); 
                    else
                      lgraph = layerGraph(net);
                    end 

                    [learnableLayer,classLayer] = findLayersToReplace(lgraph);
                    [learnableLayer,classLayer] 
                    
                    %numClasses = numel(unique(categories(trainingSet.Labels)))
                    
                    if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
                        newLearnableLayer = fullyConnectedLayer(numClasses, ...
                            'Name','new_fc', ...
                            'WeightLearnRateFactor',10, ...
                            'BiasLearnRateFactor',10);

                    elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
                        newLearnableLayer = convolution2dLayer(1,numClasses, ...
                            'Name','new_conv', ...
                            'WeightLearnRateFactor',10, ...
                            'BiasLearnRateFactor',10);
                    end

                    lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

                    newClassLayer = classificationLayer('Name','new_classoutput');
                    lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%                     figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
%                     plot(lgraph)
%                     ylim([0,10])

                    layers = lgraph.Layers;
                    connections = lgraph.Connections;
                    NumFreezeLayers=size(net.Layers(1:end-3), 1);

                    %  weight updating - non -freezed
                    layers(1:10) = freezeWeights(layers(1:10));
                    % no weight update - freezed
                    %
                    if (ExeType == 4)
                        layers(1:NumFreezeLayers) = freezeWeights(layers(1:NumFreezeLayers));
                    end
                    
                    lgraph1 = createLgraphUsingConnections(layers,connections);

                elseif strcmp(methodName(1,iq), 'alexnet')
                    net = alexnet();
                    
                    inputSize = net.Layers(1).InputSize
                    layersTransfer = net.Layers(1:end-3);

                    NumFreezeLayers=size(net.Layers(1:end-3), 1);

                    %  weight updating - non -freezed
                    layersTransfer(1:10) = freezeWeights(layersTransfer(1:10));
                    % no weight update - freezed

                    if (ExeType == 4)
                        layersTransfer(1:NumFreezeLayers) = freezeWeights(layersTransfer(1:NumFreezeLayers));
                    end
                    
                    % layersTransferAll = net;
                    
                    %numClasses = numel(unique(categories(trainingSet.Labels)))
                    
                    lgraph1 = [
                    layersTransfer
                    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
                    softmaxLayer('Name','new_softmax')
                    classificationLayer('Name','new_classi_layer')];
                elseif strcmp(methodName(1,iq), 'googlenet')
                    net = googlenet();
                    
                    if isa(net,'SeriesNetwork') 
                      lgraph = layerGraph(net.Layers); 
                    else
                      lgraph = layerGraph(net);
                    end 

                    [learnableLayer,classLayer] = findLayersToReplace(lgraph);
                    [learnableLayer,classLayer] 
                    
                    %numClasses = numel(unique(categories(trainingSet.Labels)))
                    
                    if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
                        newLearnableLayer = fullyConnectedLayer(numClasses, ...
                            'Name','new_fc', ...
                            'WeightLearnRateFactor',10, ...
                            'BiasLearnRateFactor',10);

                    elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
                        newLearnableLayer = convolution2dLayer(1,numClasses, ...
                            'Name','new_conv', ...
                            'WeightLearnRateFactor',10, ...
                            'BiasLearnRateFactor',10);
                    end

                    lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

                    newClassLayer = classificationLayer('Name','new_classoutput');
                    lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%                     figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
%                     plot(lgraph)
%                     ylim([0,10])

                    layers = lgraph.Layers;
                    connections = lgraph.Connections;
                    NumFreezeLayers=size(net.Layers(1:end-3), 1);

                    %  weight updating - non -freezed
                    layers(1:10) = freezeWeights(layers(1:10));
                    % no weight update - freezed
                    %
                    if (ExeType == 4)
                        layers(1:NumFreezeLayers) = freezeWeights(layers(1:NumFreezeLayers));
                    end
                    
                    lgraph1 = createLgraphUsingConnections(layers,connections);

                end
                
%                 pixelRange = [-10 10];
%                 imageAugmenter = imageDataAugmenter( ...
%                     'RandXReflection',true, ...
%                     'RandXTranslation',pixelRange, ...
%                     'RandYTranslation',pixelRange);
%                 augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingSet, ...
%                     'DataAugmentation',imageAugmenter);
% 
%                 augimdsValidation = augmentedImageDatastore(inputSize(1:2),testSet);

                validationFrequency = floor(size(trainingSet.Files, 1)/MinBatchS) % trainingSet.NumObservations

                options = trainingOptions('sgdm', ...
                    'MiniBatchSize',MinBatchS, ...
                    'MaxEpochs',mepo, ...
                    'InitialLearnRate',1e-4, ...
                    'Shuffle','every-epoch', ...
                    'ValidationData',valSet, ...
                    'ValidationFrequency',3, ...
                    'Verbose',true, ...
                    'VerboseFrequency',validationFrequency, ...
                    'Plots','training-progress', ...
                    'OutputFcn', @(info)savetrainingplot(info, fileSaveName));

                

                [trainedNet, info] = trainNetwork(trainingSet,lgraph1,options);
                
                YPred = classify(trainedNet,testSet);
                
                YValidation = testSet.Labels;
                %calculate accuracy
                Conf = confusionmat(YValidation,YPred)

                tp = sum((double(YPred) == 2) & (double(YValidation) == 2));
                fp = sum((double(YPred) == 2) & (double(YValidation) == 1));
                fn = sum((double(YPred) == 1) & (double(YValidation) == 2));

                accuracy = (sum(YPred == YValidation)/numel(YValidation))*100;
                                    precision = (tp / (tp + fp))*100;
                                    recall = (tp / (tp + fn))*100;
                                    F1_measure = ((2 * precision * recall) / (precision + recall));

                                    Result_Perameters= strcat("Accuracy      precision      recall      F1_measure");
                                    Test_Results= strcat(num2str(accuracy), "        ", num2str(precision), "       ", num2str(recall), "       ", num2str(F1_measure));
                                    disp(fileSaveName);
                                    disp(Result_Perameters);
                                    disp(Test_Results);


                                    delete(findall(0));
                                end
                            end
                        end
                    end

                    diary off


function stop=savetrainingplot(info, fileSaveName)
stop=false;  %prevents this function from ending trainNetwork prematurely
    if info.State=='done'   %check if all iterations have completed
    % if true
        
        exportapp(findall(groot, 'Type', 'Figure'),string(fileSaveName))  % save figure as .png, you can change this
    end
end
