% a demo code for TDD extraction

vid_name = 'test.avi';

% idt extraction
display('Extract improved trajectories...');
system(['./DenseTrackStab -f ',vid_name,' -o ',vid_name(1:end-4),'.bin']);

% TVL1 flow extraction
display('Extract TVL1 optical flow field...');
mkdir test/
system(['./denseFlow_gpu -f ',vid_name,' -x test/flow_x -y test/flow_y -b 20 -t 1 -d 3']);

% Import improved trajectories
IDT = import_idt([vid_name(1:end-4), '.bin'],15);
info = IDT.info;
tra = IDT.tra;

sizes = [8,8; 11.4286,11.4286; 16,16; 22.8571,24;32,34.2587];
sizes_vid = [480,640; 340,454; 240,320; 170,227; 120,160];

% Spatial TDD
addpath /home/lmwang/code/caffe_data_parallel/caffe/matlab
display('Extract spatial TDD...');

scale = 3;
gpu_id = 0;

model_def_file = [ 'model_proto/spatial_net_scale_', num2str(scale), '.prototxt'];
model_file = 'spatial_v2.caffemodel';

caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');
[feature_conv4, feature_conv5] = SpatialCNNFeature(vid_name, net, sizes_vid(scale,1), sizes_vid(scale,2));

if max(info(1,:)) > size(feature_conv4,4)
    ind =  info(1,:) <= size(feature_conv4,4);
    info = info(:,ind);
    tra = tra(:,ind);
end

[feature_conv_normalize_1, feature_conv_normalize_2] = FeatureMapNormalization(feature_conv4);
tdd_feature_spatial_conv4_norm_1 = TDD(info, tra, feature_conv_normalize_1, sizes(scale,1), sizes(scale,2), 1);
tdd_feature_spatial_conv4_norm_2 = TDD(info, tra, feature_conv_normalize_2, sizes(scale,1), sizes(scale,2), 1);

[feature_conv_normalize_1, feature_conv_normalize_2] = FeatureMapNormalization(feature_conv5);
tdd_feature_spatial_conv5_norm_1 = TDD(info, tra, feature_conv_normalize_1, sizes(scale,1), sizes(scale,2), 1);
tdd_feature_spatial_conv5_norm_2 = TDD(info, tra, feature_conv_normalize_2, sizes(scale,1), sizes(scale,2), 1);

% Temporal TDD
display('Extract temporal TDD...');
scale = 3;
gpu_id = 0;

model_def_file = [ 'model_proto/temporal_net_scale_', num2str(scale),'.prototxt'];
model_file = 'temporal_v2.caffemodel';

caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

[feature_conv3, feature_conv4] = TemporalCNNFeature('test/', net, sizes_vid(scale,1), sizes_vid(scale,2));
if max(info(1,:)) > size(feature_conv4,4)
    ind =  info(1,:) <= size(feature_conv4,4);
    info = info(:,ind);
    tra = tra(:,ind);
end
[feature_conv_normalize_1, feature_conv_normalize_2] = FeatureMapNormalization(feature_conv3);
tdd_feature_temporal_conv3_norm_1 = TDD(info, tra, feature_conv_normalize_1, sizes(scale,1), sizes(scale,2), 1);
tdd_feature_temporal_conv3_norm_2 = TDD(info, tra, feature_conv_normalize_2, sizes(scale,1), sizes(scale,2), 1);

[feature_conv_normalize_1, feature_conv_normalize_2] = FeatureMapNormalization(feature_conv4);
tdd_feature_temporal_conv4_norm_1 = TDD(info, tra, feature_conv_normalize_1, sizes(scale,1), sizes(scale,2), 1);
tdd_feature_temporal_conv4_norm_2 = TDD(info, tra, feature_conv_normalize_2, sizes(scale,1), sizes(scale,2), 1);
