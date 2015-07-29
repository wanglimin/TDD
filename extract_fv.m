function  extract_fv(index, power, layer, tag ,dim1, norm)


path_tdd = '/nfs/lmwang/thumos15/';
path1 = [path_tdd,'thumos15_validation_tdd_',tag,'_',layer,'_scale_1_norm_',num2str(norm),'/'];
path2 = [path_tdd,'thumos15_validation_tdd_',tag,'_',layer,'_scale_2_norm_',num2str(norm),'/'];
path3 = [path_tdd,'thumos15_validation_tdd_',tag,'_',layer,'_scale_3_norm_',num2str(norm),'/'];
path4 = [path_tdd,'thumos15_validation_tdd_',tag,'_',layer,'_scale_4_norm_',num2str(norm),'/'];
path5 = [path_tdd,'thumos15_validation_tdd_',tag,'_',layer,'_scale_5_norm_',num2str(norm),'/'];

path_tdd_fv = '/media/sdd/lmwang/Data/thumos15/';
path6 = [path_tdd_fv,'thumos15_validation_tdd_',tag,'_',layer,'_scale_1_norm_',num2str(norm),'_fv/'];
path7 = [path_tdd_fv,'thumos15_validation_tdd_',tag,'_',layer,'_scale_2_norm_',num2str(norm),'_fv/'];
path8 = [path_tdd_fv,'thumos15_validation_tdd_',tag,'_',layer,'_scale_3_norm_',num2str(norm),'_fv/'];
path9 = [path_tdd_fv,'thumos15_validation_tdd_',tag,'_',layer,'_scale_4_norm_',num2str(norm),'_fv/'];
path10 = [path_tdd_fv,'thumos15_validation_tdd_',tag,'_',layer,'_scale_5_norm_',num2str(norm),'_fv/'];


num = 256;
PCA = load(['/home/lmwang/code/tdd/ucf_pca_idt_cnn_vgg_',tag,'_',layer,'_norm_',num2str(norm),'_power_',num2str(power),'.mat']);
GMM = load(['/home/lmwang/code/tdd/ucf_gmm_',num2str(num),'_pca_',num2str(dim1),'_idt_cnn_vgg_',tag,'_',layer,'_norm_',num2str(norm),'_power_',num2str(power),'.mat']);

folderlist = dir(path3);
foldername = {folderlist(:).name};
foldername = setdiff(foldername,{'.','..'});

for i = index
    %     if ~exist([path6,foldername{i}],'dir')
    %         mkdir([path6,foldername{i}]);
    % 	end
    if ~exist([path7,foldername{i}],'dir')
        mkdir([path7,foldername{i}]);
    end
    if ~exist([path8,foldername{i}],'dir')
        mkdir([path8,foldername{i}]);
    end
    if ~exist([path9,foldername{i}],'dir')
        mkdir([path9,foldername{i}]);
    end
    if ~exist([path10,foldername{i}],'dir')
        mkdir([path10,foldername{i}]);
    end
    
    filelist = dir([path3,foldername{i},'/*.mat']);
    tic;
    for j = 1:length(filelist)
        if ~exist([path10,foldername{i},'/',filelist(j).name(1:end-4),'_pca_',num2str(dim1),'_power_',num2str(power),'.mat'],'file')
            
            %       feature = load([path1,foldername{i},'/',filelist(j).name]);
            % 		feature = double(feature.idt_cnn_feature.^power);
            %
            %         if ~isempty(feature)
            %             feature = bsxfun(@minus,feature,PCA.mu);
            %             feature = PCA.U(:,1:dim1)'*feature;
            %             feature = bsxfun(@rdivide,feature,sqrt(PCA.vars(1:dim1)));
            %             coding = vl_fisher(feature,GMM.means,GMM.covariances,GMM.priors);
            % 		else
            %
            %             coding = zeros(1,2*num*dim1);
            % 		end
            % 		save([path6,foldername{i},'/',filelist(j).name(1:end-4),'_pca_',num2str(dim1),'_power_',num2str(power),'.mat'],'coding');
            
            feature = load([path2,foldername{i},'/',filelist(j).name]);
            feature = double(feature.idt_cnn_feature.^power);
            
            if ~isempty(feature)
                feature = bsxfun(@minus,feature,PCA.mu);
                feature = PCA.U(:,1:dim1)'*feature;
                feature = bsxfun(@rdivide,feature,sqrt(PCA.vars(1:dim1)));
                coding = vl_fisher(feature,GMM.means,GMM.covariances,GMM.priors);
            else
                
                coding = zeros(1,2*num*dim1);
            end
            save([path7,foldername{i},'/',filelist(j).name(1:end-4),'_pca_',num2str(dim1),'_power_',num2str(power),'.mat'],'coding');
            
            feature = load([path3,foldername{i},'/',filelist(j).name]);
            feature = double(feature.idt_cnn_feature.^power);
            
            if ~isempty(feature)
                feature = bsxfun(@minus,feature,PCA.mu);
                feature = PCA.U(:,1:dim1)'*feature;
                feature = bsxfun(@rdivide,feature,sqrt(PCA.vars(1:dim1)));
                coding = vl_fisher(feature,GMM.means,GMM.covariances,GMM.priors);
            else
                
                coding = zeros(1,2*num*dim1);
            end
            save([path8,foldername{i},'/',filelist(j).name(1:end-4),'_pca_',num2str(dim1),'_power_',num2str(power),'.mat'],'coding');
            
            feature = load([path4,foldername{i},'/',filelist(j).name]);
            feature = double(feature.idt_cnn_feature.^power);
            
            if ~isempty(feature)
                feature = bsxfun(@minus,feature,PCA.mu);
                feature = PCA.U(:,1:dim1)'*feature;
                feature = bsxfun(@rdivide,feature,sqrt(PCA.vars(1:dim1)));
                coding = vl_fisher(feature,GMM.means,GMM.covariances,GMM.priors);
            else
                
                coding = zeros(1,2*num*dim1);
            end
            save([path9,foldername{i},'/',filelist(j).name(1:end-4),'_pca_',num2str(dim1),'_power_',num2str(power),'.mat'],'coding');
            
            feature = load([path5,foldername{i},'/',filelist(j).name]);
            feature = double(feature.idt_cnn_feature.^power);
            
            if ~isempty(feature)
                feature = bsxfun(@minus,feature,PCA.mu);
                feature = PCA.U(:,1:dim1)'*feature;
                feature = bsxfun(@rdivide,feature,sqrt(PCA.vars(1:dim1)));
                coding = vl_fisher(feature,GMM.means,GMM.covariances,GMM.priors);
            else
                
                coding = zeros(1,2*num*dim1);
            end
            save([path10,foldername{i},'/',filelist(j).name(1:end-4),'_pca_',num2str(dim1),'_power_',num2str(power),'.mat'],'coding');
        end
        
    end
    toc;
end

end

