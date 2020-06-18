import torch.nn as nn
import torch

# classification
class FMLoss(nn.Module):

  def __init__(self, weight = None, reduction = 'mean'):
    super(FMLoss, self).__init__()

  def forward(self, outputs, targets):
    '''Args:
    outputs: torch.tensor(). Size = [128, num_classes]. Use slicing to separate distillation and classification parts.
    targets: torch.tensor(). Size = [128, num_classes]. Use slicing to separate distillation and classification parts.
    '''
    num_classes = outputs.size(1) 
    
    EPS = 1e-10
    sigmoid= nn.Sigmoid()
    clf_loss = torch.mean(-targets[:, :num_classes-10]*torch.log(sigmoid(outputs[:, num_classes-10:])+EPS)\
                        + (1-targets[:, num_classes-10:])* torch.pow(sigmoid(outputs[:, num_classes-10:]), 2))
    
    if num_classes == 10:
      return clf_loss
    
    dist_criterion = nn.BCEWithLogitsLoss()
    dist_loss = dist_criterion(outputs[:, :num_classes-10], targets[:, :num_classes-10])
    
    dist = (num_classes - 10)/num_classes
    clf = 10/num_classes
    
    loss = clf*clf_loss + dist*dist_loss
    return loss
  
# Old outputs targets are given BCE likewise
# Implementation of https://arxiv.org/abs/1503.02531
class DKHLoss(nn.Module):
  '''
  Distillation Knowledge Hinton Loss
  '''
  
  def __init__(self, weight = None, reduction = 'mean', temperature = 2):
    self.T = temperature # 2 by default as in LWF paper's implementation
    super(DKHLoss, self).__init__()
    
  def forward(self, outputs, targets):
    """ Args:
        outputs = new net outputs on old classes
        targets = old_ne outputs on old_classes

        Computes the distillation loss (cross-entropy).
        xentropy(y, t) = kl_div(y, t) + entropy(t)
        entropy(t) does not contribute to gradient wrt y, so we skip that.
        Thus, loss value is slightly different, but gradients are correct.
        \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
        scale is required as kl_div normalizes by nelements and not batch size.
    """
    softmax = nn.Softmax(dim=0)
    log_softmax = nn.LogSoftmax(dim=0)

    loss = torch.mean(-softmax(targets/self.T)*torch.log(softmax(outputs/self.T)))

    # Try different T value to see how values changes
    
    return loss
    
 
# classification
class LFCLoss(nn.Module):

  def __init__(self, weight = None, reduction = 'mean'):
    super(LFCLoss, self).__init__()

  def forward(self, new_outputs, new_targets, old_features, new_features, num_classes):
    '''Args:
    outputs: torch.tensor(). Size = [64, num_classes]. Use slicing to separate distillation and classification parts.
    targets: torch.tensor(). Size = [64, num_classes]. Use slicing to separate distillation and classification parts.
    '''
    
    BATCH_SIZE = 64
    
    lambda_base = 5 # from paper
    cur_lambda = lambda_base * sqrt(num_classes-10/num_classes) # from paper
    
#     EPS = 1e-10
#     sigmoid= nn.Sigmoid()
#     clf_loss = torch.mean(-new_targets[:, :num_classes-10]*torch.log(sigmoid(outputs[:, num_classes-10:])+EPS)\
#                         + (1-new_targets[:, num_classes-10:])* torch.pow(sigmoid(outputs[:, num_classes-10:]), 2))
 
    clf_criterion = nn.BCEWithLogitsLoss()
    clf_loss = clf_criterion(new_outputs, new_targets)
    
    if num_classes == 10:
      return clf_loss
    
    dist_criterion = nn.CosineEmbeddingLoss()
    dist_loss = dist_criterion(new_features, old_features, torch.ones(BATCH_SIZE).cuda())
    
    dist = (num_classes - 10)/num_classes
    clf = 10/num_classes
    
    loss = clf*clf_loss + dist*dist_loss*cur_lambda
    
    return loss
    




# distillation - are all contributes needed? randomly remove some contributions to the loss
class MaskBCELoss(nn.Module):
  
  def __init__(self):
    super(MyBCELoss, self).__init__()

  def forward(self, outputs, targets):

    num_classes = outputs.size(1) 

    clf_criterion = nn.BCEWithLogitsLoss()

    clf_loss = clf_criterion(outputs[:, num_classes-10:], targets[:, num_classes-10:])

    if num_classes == 10:
      return clf_loss

    fraction = 0.7 # fraction of non zero entries

    sigmoid = nn.Sigmoid()
    EPS = 1e-10
    rows, cols = outputs[:, :num_classes-10].size() # [batch size, num_classes]
    size = int(rows*cols*fraction) 
    random_vector = np.zeros(int(rows*cols), dtype=int)
    random_vector[:size] = 1
    np.random.shuffle(random_vector)
    random_tensor = torch.FloatTensor(random_vector.reshape(rows, cols)).cuda()
    dist_loss = torch.mean(-random_tensor*(targets[:, :num_classes-10]*torch.log(sigmoid(outputs[:, :num_classes-10])+EPS)\
                                      + (1-targets[:, :num_classes-10])* torch.log(1 - sigmoid(outputs[:, :num_classes-10])+EPS)))
    
    dist = (num_classes - 10)/num_classes
    clf = 10/num_classes
    
    loss = clf*clf_loss + dist*dist_loss

    return loss
    
    
# distillation
class DFMLoss(nn.Module): 
  def __init__(self, weight = None, reduction = 'mean'):
    super(DFMLoss, self).__init__()
    self.reduction = reduction
    self.weight = weight

  def forward(self, outputs, targets):
    sigmoid= nn.Sigmoid()
    BETA = 0.3
    loss = torch.mean(BETA*((2*sigmoid(outputs)-1)*(torch.pow((2*sigmoid(outputs)-1), 3) -4*torch.pow((2*targets-1), 3)) +3))
    return loss
    
# distillation 
class DFMSquaredLoss(nn.Module):
  def __init__(self, weight = None, reduction = 'mean'):
    super(DFMSquaredLoss, self).__init__()
    self.reduction = reduction
    self.weight = weight

  def forward(self, outputs, targets):
    EPS = 1e-10
    sigmoid= nn.Sigmoid()
    BETA = 3
    loss = torch.mean(BETA*(targets*torch.pow((1-sigmoid(outputs)), 2) +(1-targets)*torch.pow(sigmoid(outputs), 2)))
    return loss
    
# distillation  
class BCEHarshLoss(nn.Module):
  def __init__(self, weight = None, reduction = 'mean'):
    super(BCEHarshLoss, self).__init__()
    self.reduction = reduction
    self.weight = weight

  def forward(self, outputs, targets):
    EPS = 1e-10
    sigmoid= nn.Sigmoid()
    BETA = 0.1
    squared_distance = torch.pow((outputs-targets), 2)
    loss = torch.mean(-BETA*torch.log(1-squared_distance) -(targets*torch.log(sigmoid(outputs)+EPS) + (1-targets)* torch.log(1 - sigmoid(outputs)+EPS)))
    return loss

# distillation
class BCESquaredLoss(nn.Module):
  pass

# distillation
class BCELineLoss(nn.Module):
  pass
