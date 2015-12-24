function extract_tdd(layer, scale, gpu_id, tag, ind)

if strcmp(tag,'flow')
    path_tra = '../thumos15_validation_tra/';
    path_vid = '../thumos15_validation_clips/';
    path_flow = '/home/lmwang/Data/THUMOS/thumos15_validation_flow/';
    path1 = ['../thumos15_validation_tdd_flow_',layer,'_scale_',num2str(scale),'_norm_2/'];
    path2 = ['../thumos15_validation_tdd_flow_',layer,'_scale_',num2str(scale),'_norm_3/'];
    model_def_file = [ 'models/flow_',layer,'_scale',num2str(scale),'.prototxt'];
    model_file = '10_flow_iter_90000.caffemodel';
    sizes_vid = [480,640; 340,454; 240,320; 170,227; 120,160];
    
    folderlist = dir(path_vid);
    foldername = {folderlist(:).name};
    foldername = setdiff(foldername,{'.','..'});
    
    for i = ind
        i
        if ~exist([path1,foldername{i}],'dir')
            mkdir([path1,foldername{i}]);
        end
        if ~exist([path2,foldername{i}],'dir')
            mkdir([path2,foldername{i}]);
        end
        
        filelist = dir([path_vid,foldername{i},'/*.avi']);
        
        for j = 1:length(filelist)
            
            if ~exist([path2,foldername{i},'/',filelist(j).name(1:end-4),'.mat'],'file')
                data = load([path_tra,foldername{i},'/',filelist(j).name(1:end-4),'.mat']);
                info = data.info;
                tra = data.tra;
                if  ~isempty(info)
                    tic;
                    filename = [path_flow,foldername{i},'/',filelist(j).name(1:end-4),'/'];
                    feature = FlowCNNFeature(filename, 1, sizes_vid(scale,1), sizes_vid(scale,2),model_def_file, model_file, gpu_id);
                    sizes = [8,8; 11.4286,11.4286; 16,16; 22.8571,24;32,34.2587];
                    if max(info(1,:)) > size(feature,4)
                        max(info(1,:))
                        size(feature,4)
                        ind =  info(1,:) <= size(feature,4);
                        info = info(:,ind);
                        tra = tra(:,ind);
                    end
                    [cnn_feature1, cnn_feature2] = FeatureMapNormalization(feature);
                    idt_cnn_feature = IDTCNN2(info, tra, cnn_feature1, sizes(scale,1), sizes(scale,2), 1);
                    save([path1,foldername{i},'/',filelist(j).name(1:end-4),'.mat'],'idt_cnn_feature','-v7.3');
                    idt_cnn_feature = IDTCNN2(info, tra, cnn_feature2, sizes(scale,1), sizes(scale,2), 1);
                    save([path2,foldername{i},'/',filelist(j).name(1:end-4),'.mat'],'idt_cnn_feature','-v7.3');
                    toc;
                end
            end
        end
    end
end

if strcmp(tag,'rgb')
    path_tra = '../thumos15_validation_tra/';
    path_vid = '/home/lmwang/Data/THUMOS/thumos15_validation_clips/';
    path1 = ['../thumos15_validation_tdd_rgb_',layer,'_scale_',num2str(scale),'_norm_2/'];
    path2 = ['../thumos15_validation_tdd_rgb_',layer,'_scale_',num2str(scale),'_norm_3/'];
    model_def_file = [ 'models/rgb_',layer,'_scale',num2str(scale),'.prototxt'];
    model_file = 'spatial.caffemodel';
    sizes_vid = [480,640; 340,454; 240,320; 170,227; 120,160];
    
    folderlist = dir(path_vid);
    foldername = {folderlist(:).name};
    foldername = setdiff(foldername,{'.','..'});
    
    for i = ind
        i
        if ~exist([path1,foldername{i}],'dir')
            mkdir([path1,foldername{i}]);
        end
        if ~exist([path2,foldername{i}],'dir')
            mkdir([path2,foldername{i}]);
        end
        
        filelist = dir([path_vid,foldername{i},'/*.avi']);
        for j = 1:length(filelist)
            
            if  ~exist([path2,foldername{i},'/',filelist(j).name(1:end-4),'.mat'],'file')
                data = load([path_tra,foldername{i},'/',filelist(j).name(1:end-4),'.mat']);
                info = data.info;
                tra = data.tra;
                if ~isempty(info)
                    tic;
                    filename = [path_vid,foldername{i},'/',filelist(j).name];
                    feature = RGBCNNFeature(filename, 1, sizes_vid(scale,1), sizes_vid(scale,2), model_def_file, model_file, gpu_id);
                    sizes = [8,8; 11.4286,11.4286; 16,16; 22.8571,24;32,34.2587];
                    if max(info(1,:)) > size(feature,4)
                        max(info(1,:))
                        size(feature,4)
                        ind =  info(1,:) <= size(feature,4);
                        info = info(:,ind);
                        tra = tra(:,ind);
                    end
                    [cnn_feature1, cnn_feature2] = FeatureMapNormalization(feature);
                    idt_cnn_feature = TDD(info, tra, cnn_feature1, sizes(scale,1), sizes(scale,2), 1);
                    save([path1,foldername{i},'/',filelist(j).name(1:end-4),'.mat'],'idt_cnn_feature','-v7.3');
                    idt_cnn_feature = TDD(info, tra, cnn_feature2, sizes(scale,1), sizes(scale,2), 1);
                    save([path2,foldername{i},'/',filelist(j).name(1:end-4),'.mat'],'idt_cnn_feature','-v7.3');
                    toc;
                end
            end
        end
    end
end
end