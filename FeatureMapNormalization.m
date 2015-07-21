function [cnn_feature1, cnn_feature2] = FeatureMapNormalization(cnn_feature)

r = size(cnn_feature,1);
c = size(cnn_feature,2);
f = size(cnn_feature,3);
t = size(cnn_feature,4);
cnn_feature1 = permute(cnn_feature,[1,2,4,3]);
cnn_feature1 = reshape(cnn_feature1,r*c*t,[]);
cnn_feature1 = bsxfun(@rdivide,cnn_feature1,max(cnn_feature1,[],1)+eps);
cnn_feature1 = reshape(cnn_feature1,r,c,t,f); 
cnn_feature1 = permute(cnn_feature1,[1,2,4,3]);
cnn_feature2 = bsxfun(@rdivide,cnn_feature,max(cnn_feature,[],3)+eps);

end