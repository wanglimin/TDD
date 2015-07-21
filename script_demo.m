% a demo code for TDD extraction

vid_name = 'test.avi';

% idt extraction
display('Extract improved trajectories...');
system(['./DenseTrackStab -f ',vid_name,' -o ',vid_name(1:end-4),'.bin']);

% TVL1 flow extraction
display('Extract TVL1 optical flow field...');
mkdir test/
system(['./dense_flow -f ',vid_name,' -x test/flow_x -y test/flow_y -b 20 -t 1 -d 3']);

% Import improved trajectories
IDT = import_idt('test.bin',15);
info = IDT.info;
tra = IDT.tra;

sizes = [8,8; 11.4286,11.4286; 16,16; 22.8571,24;32,34.2587];
sizes_vid = [480,640; 340,454; 240,320; 170,227; 120,160];

% Spatial TDD
display('Extract spatial TDD...');

scale = 3;
layer = 'conv5';
gpu_id = 0;

model_def_file = [ 'models/rgb_',layer,'_scale',num2str(scale),'.prototxt'];
model_file = 'spatial.caffemodel';

feature_conv = RGBCNNFeature(vid_name, 1, sizes_vid(scale,1), sizes_vid(scale,2), model_def_file, model_file, gpu_id);
if max(info(1,:)) > size(feature_conv,4)
    ind =  info(1,:) <= size(feature_conv,4);
    info = info(:,ind);
    tra = tra(:,ind);
end

[feature_conv_normalize_1, feature_conv_normalize_2] = FeatureMapNormalization(feature_conv);
tdd_feature_spatial_1 = TDD(info, tra, feature_conv_normalize_1, sizes(scale,1), sizes(scale,2), 1);
tdd_feature_spatial_2 = TDD(info, tra, feature_conv_normalize_2, sizes(scale,1), sizes(scale,2), 1);

% Temporal TDD
display('Extract temporal TDD...');
scale = 3;
layer = 'conv5';
gpu_id = 0;

model_def_file = [ 'models/flow_',layer,'_scale',num2str(scale),'.prototxt'];
model_file = 'temporal.caffemodel';

caffe('reset');
feature_conv = FlowCNNFeature('test/', 1, sizes_vid(scale,1), sizes_vid(scale,2),model_def_file, model_file, gpu_id);
if max(info(1,:)) > size(feature_conv,4)
    ind =  info(1,:) <= size(feature_conv,4);
    info = info(:,ind);
    tra = tra(:,ind);
end
[feature_conv_normalize_1, feature_conv_normalize_2] = FeatureMapNormalization(feature_conv);
tdd_feature_temporal_1 = TDD(info, tra, feature_conv_normalize_1, sizes(scale,1), sizes(scale,2), 1);
tdd_feature_temporal_2 = TDD(info, tra, feature_conv_normalize_2, sizes(scale,1), sizes(scale,2), 1);
