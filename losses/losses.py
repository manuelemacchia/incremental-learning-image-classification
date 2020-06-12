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
    dist_criterion = nn.BCEWithLogitsLoss()
    dist_loss = dist_criterion(outputs[:, :num_classes-10], targets[:, :num_classes-10])
    
    EPS = 1e-10
    sigmoid= nn.Sigmoid()
    clf_loss = torch.mean(-targets*torch.log(sigmoid(outputs[:, num_classes-10:])+EPS)\
                        + (1-targets[:, num_classes-10:])* torch.pow(sigmoid(outputs[:, num_classes-10:]), 2))
    
    dist = (self.num_classes - 10)/self.num_classes
    clf = 10/self.num_classes
    
    loss = clf*clf_loss + dist*dist_loss
    return loss
  
  
# CE-style targets (no one hot encoding)
# Implementation of https://arxiv.org/abs/1503.02531
class DKHLoss(nn.Module):
  '''
  Distillation Knowledge Hinton Loss
  '''
  
  def __init__(self, weight = None, reduction = 'mean', temperature = 2):
    self.T = temperature # 2 by default as in LWF paper's implementation
    super(HintonLoss, self).__init__()
    
  def forward(self, outputs, targets):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """
    # outputs.size() = [batch_size, num_classes]
    # targets.size() = [batch_size, 1]
    
   def forward(outputs, targets):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """
    # outputs.size() = [batch_size, num_classes]
    # targets.size() = [batch_size, 1]
    
    old_outputs = outputs[:, :-10] # old_classes outputs
    old_targets = targets[:, :-10] # old_net_outputs
    
    new_outputs = outputs[:, -10:]
    one_hot_labels = targets[:, -10:]

    softmax = nn.Softmax() # softmax(targets)
    log_softmax = nn.LogSoftmax()
    
    clf_loss = torch.sum(-one_hot_labels*torch.log(softmax(new_outputs)), dim = 0)
    dist_loss = torch.sum(-softmax(old_targets/2)*(log_softmax(old_outputs/2)), dim =0)
    
    loss = torch.mean(torch.cat((clf_loss, dist_loss),dim = 0), dim= 0)
    
    return loss
    
    

# distillation - are all contributes needed? randomly remove some contributions to the loss
class MyBCELoss(nn.Module):
  
  def __init__(self):
    super(MyBCELoss, self).__init__()

  def forward(self, outputs, targets, random_tensor):
    sigmoid = nn.Sigmoid()
    EPS = 1e-10
    rows, cols = outputs.size()
    size = int(rows*cols*0.9)
    random_vector = np.zeros(int(rows*cols), dtype=int)
    random_vector[:size] = 1
    np.random.shuffle(random_vector)
    random_tensor = torch.FloatTensor(random_vector.reshape(rows, cols)).cuda()
    loss = torch.mean(-random_tensor*(targets*torch.log(sigmoid(outputs)+EPS) + (1-targets)* torch.log(1 - sigmoid(outputs)+EPS)))

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
