def LfC_loss(self,old_outputs,new_features,new_output,labels,step,current_step,utils,eta,lambda_base = 2,n_classes=10,batch_size=128,m=0.5,K=2):
        n_old_classes = n_classes*(step-1)
        clf_criterion = nn.CrossEntropyLoss(reduction = 'mean')
        cosine_loss = nn.CosineEmbeddingLoss(reduction='mean')
        if step == 1 or current_step==-1:
            clf_loss = clf_criterion(new_output,labels)
            return clf_loss
        
        #Classification
        clf_loss = clf_criterion(new_output,labels)
        
        #Distillation
        lambda_dist = lambda_base*((n_classes/n_old_classes)**0.5)
        dist_loss = lambda_dist*cosine_loss(new_features, old_outputs,torch.ones(batch_size).cuda()) 

        #Margin
        exemplar_idx = sum(labels.cpu().numpy() == label for label in range(n_old_classes)).astype(bool)
        exemplar_labels = labels[exemplar_idx].type(torch.long)
        anchors = new_output[exemplar_idx, exemplar_labels] / eta
        out_new_classes = new_output[exemplar_idx, n_old_classes:] / eta
        topK_hard_negatives, _ = torch.topk(out_new_classes, K)
        loss_mr = torch.max(m - anchors.unsqueeze(1).cuda() + topK_hard_negatives.cuda(), torch.zeros(1).cuda()).sum(dim=1).mean()


        return clf_loss + dist_loss + loss_mr
