function [feature] = TDD(inf,tra,cnn_feature,scale_x,scale_y,num_cell)

if ~isempty(inf)
	ind = inf(7,:)==1;
	inf = inf(:,ind);
	tra = tra(:,ind);
end


if ~isempty(inf)
	NUM_DIM = size(cnn_feature,3);
	NUM_DES = size(inf,2);
	TRA_LEN = size(tra,1)/2;

	num_fea = TRA_LEN / num_cell;

	pos = reshape(tra,2,[])-1;
	pos = round(bsxfun(@rdivide,pos,[scale_x;scale_y]) + 1);
	pos = bsxfun(@max,pos,[1;1]);
	pos = bsxfun(@min,pos,[size(cnn_feature,2);size(cnn_feature,1)]);
	pos = reshape(pos,TRA_LEN*2,[]);

	cnn_feature = permute(cnn_feature,[1,2,4,3]);
	offset = [TRA_LEN-1:-1:0];
	size_mat = [size(cnn_feature,1),size(cnn_feature,2),size(cnn_feature,3)];
	cnn_feature = reshape(cnn_feature,[],NUM_DIM);

	cur_x = pos(1:2:end,:);
	cur_y = pos(2:2:end,:);
	cur_t = bsxfun(@minus,inf(1,:),offset');

	tmp = cnn_feature(sub2ind(size_mat,cur_y,cur_x,cur_t),:)';
	tmp = reshape(tmp,NUM_DIM,num_fea,[]);
	feature = reshape(sum(tmp,2),[],NUM_DES);
else
	feature = [];
end


end