function FCNNFeature = FlowCNNFeature(vid_name, use_gpu, NUM_HEIGHT, NUM_WIDTH, model_def_file, model_file, gpu_id)

L = 10;
% Input video
filelist =dir([vid_name,'*_x*.jpg']);
if length(filelist) > 30 *60
    video = zeros(NUM_HEIGHT,NUM_WIDTH,L*2,30*60,'single');
else
    video = zeros(NUM_HEIGHT,NUM_WIDTH,L*2,length(filelist),'single');
end

for i = 1:size(video,4)
    flow_x = imread(sprintf('%s_%04d.jpg',[vid_name,'flow_x'],i));
    flow_y = imread(sprintf('%s_%04d.jpg',[vid_name,'flow_y'],i));
    video(:,:,1,i) = imresize(flow_x,[NUM_HEIGHT,NUM_WIDTH],'bilinear');
    video(:,:,2,i) = imresize(flow_y,[NUM_HEIGHT,NUM_WIDTH],'bilinear');
end

for i = 1:L-1
    tmp = cat(4, video(:,:,(i-1)*2+1:i*2,2:end),video(:,:,(i-1)*2+1:i*2,end));
    video(:,:,i*2+1:i*2+2,:)  = tmp;
end

% Initialize ConvNet
if caffe('is_initialized') == 0
	if exist('use_gpu', 'var')
		matcaffe_init(use_gpu,model_def_file,model_file,gpu_id);
	else
		matcaffe_init();
	end
end


% Computing convolutional maps
d  = load('flow_mean');
FLOW_MEAN = d.image_mean;
FLOW_MEAN = imresize(FLOW_MEAN,[NUM_HEIGHT,NUM_WIDTH]);
video = bsxfun(@minus,video,FLOW_MEAN);
video = permute(video,[2,1,3,4]);

batch_size = 40;
num_images = size(video,4);
num_batches = ceil(num_images/batch_size);
FCNNFeature = [];

images = zeros(NUM_WIDTH, NUM_HEIGHT, L*2, batch_size, 'single');
for bb = 1 : num_batches
    range = 1 + batch_size*(bb-1): min(num_images,batch_size*bb);
    tmp = video(:,:,:,range);
    images(:,:,:,1:size(tmp,4)) = tmp;
	
    feature = caffe('forward',{images});
    feature = permute(feature{1},[2,1,3,4]);
    if isempty(FCNNFeature)
        FCNNFeature = zeros(size(feature,1), size(feature,2), size(feature,3), num_images, 'single');
    end
    FCNNFeature(:,:,:,range) = feature(:,:,:,mod(range-1,batch_size)+1);
end

end