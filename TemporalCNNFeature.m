function [FCNNFeature_c4, FCNNFeature_c3] = TemporalCNNFeature(vid_name, net, NUM_HEIGHT, NUM_WIDTH)

L = 10;
% Input video
filelist =dir([vid_name,'*_x*.jpg']);
video = zeros(NUM_HEIGHT,NUM_WIDTH,L*2,length(filelist));
for i = 1: length(filelist)
    flow_x = imread(sprintf('%s_%04d.jpg',[vid_name,'flow_x'],i));
    flow_y = imread(sprintf('%s_%04d.jpg',[vid_name,'flow_y'],i));
    video(:,:,1,i) = imresize(flow_x,[NUM_HEIGHT,NUM_WIDTH],'bilinear');
    video(:,:,2,i) = imresize(flow_y,[NUM_HEIGHT,NUM_WIDTH],'bilinear');
end

for i = 1:L-1
    tmp = cat(4, video(:,:,(i-1)*2+1:i*2,2:end),video(:,:,(i-1)*2+1:i*2,end));
    video(:,:,i*2+1:i*2+2,:)  = tmp;
end

d  = load('flow_mean');
FLOW_MEAN = d.image_mean;
FLOW_MEAN = imresize(FLOW_MEAN,[NUM_HEIGHT,NUM_WIDTH]);

batch_size = 40;
num_images = size(video,4);
num_batches = ceil(num_images/batch_size);

FCNNFeature_c4 = [];
FCNNFeature_c3 = [];

for bb = 1 : num_batches
    range = 1 + batch_size*(bb-1): min(num_images,batch_size*bb);
    images = zeros(NUM_WIDTH, NUM_HEIGHT, L*2, batch_size, 'single');
    tmp = video(:,:,:,range);
    
    for ii = 1 : size(tmp,4)
        img = single(tmp(:,:,:,ii));
        images(:,:,:,ii) = permute(img -FLOW_MEAN,[2,1,3]);
    end
    
    net.blobs('data').set_data(images);
    net.forward_prefilled();
    feature_c4 = permute(net.blobs('conv4').get_data(),[2,1,3,4]);
    feature_c3 = permute(net.blobs('conv3').get_data(),[2,1,3,4]);
    
    if isempty(FCNNFeature_c4)
        FCNNFeature_c4 = zeros(size(feature_c4,1), size(feature_c4,2), size(feature_c4,3), num_images, 'single');
        FCNNFeature_c3 = zeros(size(feature_c3,1), size(feature_c3,2), size(feature_c3,3), num_images, 'single');
    end
    FCNNFeature_c4(:,:,:,range) = feature_c4(:,:,:,mod(range-1,batch_size)+1);
    FCNNFeature_c3(:,:,:,range) = feature_c3(:,:,:,mod(range-1,batch_size)+1);   
end

end