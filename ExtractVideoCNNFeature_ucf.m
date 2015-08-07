function ExtractVideoCNNFeature_ucf(layer,scale,gpu_id, tag, ind)
addpath /home/lmwang/code/caffe_data_parallel/caffe/matlab
sizes =[480,640; 340,454; 240,320; 170,227; 120,160];
path_org = '/nfs/lmwang/lmwang/Data/UCF101/ucf101_org/';
folderlist = dir(path_org);
foldername = {folderlist(:).name};
foldername = setdiff(foldername,{'.','..'});

% FLOW
if  strcmp(tag,'flow') 
    path2 = ['/media/RAID0/lmwang/data/UCF/ucf101_tvl1_flow_conv5_scale_',num2str(scale),'/'];
    path3 = ['/media/RAID0/lmwang/data/UCF/ucf101_tvl1_flow_conv4_scale_',num2str(scale),'/'];
    path4 = ['/media/RAID0/lmwang/data/UCF/ucf101_tvl1_flow_conv3_scale_',num2str(scale),'/'];
    path5 = ['/media/RAID0/lmwang/data/UCF/ucf101_tvl1_flow_pool2_scale_',num2str(scale),'/'];
    path_flow =  '/media/RAID0/lmwang/data/UCF/ucf101_flow_img_tvl1_gpu/';
    
    model_def_file = [ 'models/flow_',layer,'_scale_',num2str(scale),'_new.prototxt'];
    model_file = '10_flow_iter_90000_new.caffemodel';
    caffe.reset_all();
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
    net = caffe.Net(model_def_file, model_file, 'test');

    for i = ind
        if ~exist([path2,foldername{i}],'dir')
            mkdir([path2,foldername{i}]);
        end
        if ~exist([path3,foldername{i}],'dir')
            mkdir([path3,foldername{i}]);
        end
        if ~exist([path4,foldername{i}],'dir')
            mkdir([path4,foldername{i}]);
        end
        if ~exist([path5,foldername{i}],'dir')
            mkdir([path5,foldername{i}]);
        end
        filelist = dir([path_org,foldername{i},'/*.avi']);
        for j = 1:length(filelist)
            filename = [path_flow,foldername{i},'/',filelist(j).name(1:end-4),'/'];
            if  ~exist([path5,foldername{i},'/',filelist(j).name(1:end-4),'.mat'],'file')
                tic;
                [feature_c5, feature_c4, feature_c3, feature_p2] = TemporalCNNFeature(filename, net, sizes(scale,1), sizes(scale,2));
                toc;
                tic;
                feature = feature_c5;
                save([path2,foldername{i},'/',filelist(j).name(1:end-4),'.mat'],'feature','-v7.3');
                feature = feature_c4;
                save([path3,foldername{i},'/',filelist(j).name(1:end-4),'.mat'],'feature','-v7.3');
                feature = feature_c3;
                save([path4,foldername{i},'/',filelist(j).name(1:end-4),'.mat'],'feature','-v7.3');
                feature = feature_p2;
                save([path5,foldername{i},'/',filelist(j).name(1:end-4),'.mat'],'feature','-v7.3');
                toc;
            end
        end
        i
    end
    caffe.reset_all();
end

