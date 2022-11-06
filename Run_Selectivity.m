% Result 1: Emergence of selectivity to various objects in untrained networks
 
figure('units','normalized','outerposition',[0 0.2 1 0.8]);
sgtitle('Result 1 : Emergence of selectivity to various objects in untrained networks')
 
%% Low-level feature-controlled stimulus (Figure 1A, Supple Figure 1)
stimulus_idx = [1:5,11+1:11+5];
for ii=1:10
    subplot(4,11,stimulus_idx(ii));
    imagesc(IMG_ORI(:,:,1,numIMG*(ii-1)+1));
    colormap('gray');
    axis square;
    axis off;
    title(STR_LABEL(ii),'Interpreter','none');
end

net_rand = Cell_Net{1};  
 
%% Object-selective response (Figure 1C, left)
num_cell = prod(array_sz(layerArray(5),:));
net_rand = Cell_Net{1}; Idx_Target = Cell_Idx{1,5};
 
act_rand = activations(net_rand,IMG_ORI,layersSet{layerArray(5)});
[rep_mat,rep_mat_3D] = fun_ResZscore(act_rand,num_cell,Idx_Target,numCLS,numIMG);
 
ax2 = subplot(4,11,[7:9,11+7:11+9]);
load('Colorbar_Tsao.mat');
imagesc(rep_mat);
caxis([-3 3])
for cc = 1:numCLS-1
line([numIMG*cc numIMG*cc], [1 length(Idx_Target)],'color','k','LineStyle','--')
end
xticks([numIMG/2:numIMG:numIMG*(numCLS+0.5)]); xticklabels(STR_LABEL);
c = colorbar; colormap(ax2,cmap);
c.Label.String = 'Response (z-scored)';
ylabel('Unit indices'); title('Responses of object-selective units'); clearvars cmap
set(gca,'TickLabelInterpreter','none','TickDir','out');
 
%% Object-selectivity index
rep_shuf_mat_3D = reshape(rep_mat_3D(randperm(numel(rep_mat_3D))),size(rep_mat_3D));
osi_mat = fun_OSI(rep_mat_3D);
osi_shuf_mat = fun_OSI(rep_shuf_mat_3D);
 
subplot(4,11,[2*11+4:2*11+6, 3*11+4:3*11+6]); hold on;
boxplot([osi_mat,osi_shuf_mat])
xticks([1:2]); xticklabels({'Untrained','Response shuffled'}); xlim([0.5 2.5]);
ylabel('Object-selectivity index'); title('Single neuron tuning ');
set(gca,'TickLabelInterpreter','none','TickDir','out');
 
%% Single unit tuning curve (Figure 1C, right)
subplot(4,11,[10,11,11+10,11+11]); hold on;
max_osi_idx = find(osi_mat == max(osi_mat));
plot([0:numCLS+1],[0 mean(rep_mat_3D(max_osi_idx,:,:),3) 0],'color','r');
xlim([0.5 numCLS+0.5]); xticks([1:numCLS]); xticklabels(STR_LABEL);
yline(0, '--');
ylabel('Response (z-scored)'); title(['Tuning curve of object-selective units (#' num2str(max_osi_idx) ' unit)']);
set(gca,'TickLabelInterpreter','none','TickDir','out');
 
%% Averaging tuning curve (Figure 1D)
STR_BASIC =  {'toilet', 'bed','chair','desk','dresser','monitor', ...
        'night_stand', 'sofa', 'table', 'scrambled'};
    
subplot(4,11,[2*11+1:2*11+3, 3*11+1:3*11+3]); hold on;
shadedErrorBar([0:numCLS+1],[0 mean(mean(rep_mat_3D(:,:,:),3),1) 0],...
    [0 std(mean(rep_mat_3D(:,:,:),3),0,1) 0])
xlim([0.5 numCLS+0.5]); xticks([1:numCLS]); xticklabels(STR_LABEL);
ylim([-1 1]);
yline(0, '--');
ylabel('Response (z-scored)'); title('Averaged tuning curve of object-selective units');
set(gca,'TickLabelInterpreter','none','TickDir','out');
 
%% Clustering in latent space (t-SNE) (Figure 1F, Supple Figure 2)
 
order_ClsIMG = [1 2 3 4 5 6 7 8 9];                                        
 
InitialY = 1e-4*randn(size(IMG_ORI,4)-200,2); Perplexity = 50;
labels = []; for cc = 1:numCLS-1; labels = [labels;cc.*ones(numIMG,1)];end
cmap = flip(jet(numCLS-1+3)); cmap = cmap(round(linspace(1,numCLS-1+3,numCLS-1+1)),:);
cmap = cmap(2:end,:); cmap = cmap(order_ClsIMG,:); sz = 10;
 
actIMG = reshape(IMG_ORI(:,:,1,1:1800),prod(size(IMG_ORI,1:2)),size(IMG_ORI,4)-200); % image
tSNE_IMG = tsne(actIMG','InitialY',InitialY,'Perplexity',Perplexity,'Standardize',0);  
 
layer_idx = 5;                                                             % conv5
num_cell = prod(array_sz(layer_idx,:));
act_rand = activations(net_rand,IMG_ORI(:,:,:,1:1800),layersSet{layer_idx});
act = reshape(act_rand,num_cell,size(IMG_ORI,4)-200);                      % network response
tSNE_Resp = tsne(act','InitialY',InitialY,'Perplexity',Perplexity,'Standardize',0);

subplot(4,11,[2*11+8,2*11+9,3*11+8,3*11+9]); hold on
for ii = numCLS-1:-1:1
    idx = find(double(labels) == ii);
    h = gscatter(tSNE_IMG(idx,1),tSNE_IMG(idx,2),labels(idx),cmap(ii,:),'.',sz,'off');
end

ylabel('tSNE axis 2'); xlim([-50 50]); ylim([-50 50]); title('Raw images (t-SNE)');
 
subplot(4,11,[2*11+10,2*11+11,3*11+10,3*11+11]); hold on
for ii = numCLS-1:-1:1
    idx = find(double(labels) == ii);
    h = gscatter(tSNE_Resp(idx,1),tSNE_Resp(idx,2),labels(idx),cmap(ii,:),'.',sz,'off');
end
legend(STR_LABEL(1:numCLS-1));
xlabel('tSNE axis 1'); ylabel('tSNE axis 2'); xlim([-50 50]); ylim([-50 50]); title('Conv response in untrained networks (t-SNE)');
