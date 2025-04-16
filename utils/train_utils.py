import time
import torch
import sys, os
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

def compute_f1_auc(y_true, y_pred, y_prob):
    # Compute macro-average F1 score
    f1 = f1_score(y_true, y_pred, average='macro')
    # Compute multi-class AUC using one-vs-rest approach.
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except Exception:
        auc = 0.0
    return f1, auc

def get_layer_names(model):
    """Extract meaningful layer names from model"""
    layer_names = []
    for name, _ in model.named_parameters():
        # Create a simplified name by keeping only the primary component
        if '.' in name:
            layer_name = name.split('.')[0]
            if layer_name not in layer_names:
                layer_names.append(layer_name)
        else:
            if name not in layer_names:
                layer_names.append(name)
    return layer_names

def compute_gradient_norms(model, layer_names):
    """Compute gradient norms for each layer"""
    gradient_norms = {layer: [] for layer in layer_names}  # Initialize as lists
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer = name.split('.')[0] if '.' in name else name
            if layer in gradient_norms:
                gradient_norms[layer].append(param.grad.norm().item())  # Append norm
    
    return gradient_norms

def run_training(model, train_loader, optimizer, criterion, device, epochs, scheduler=None, layer_names=None):
    """
    Generic training loop with support for learning rate schedulers
    """
    metrics = {'epoch': [], 'loss': [], 'accuracy': [], 'f1_score': []}
    gradient_norms_history = {layer: [] for layer in layer_names} if layer_names else {}  # Initialize history
    epoch_times = []  # Record each epoch's duration
    training_start = time.time()
    
    iter_costs = []  # Store loss per iteration
    iter_times = []   # Store time per iteration
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []  # Add container for probability scores
        
        for inputs, targets in train_loader:
            iter_start_time = time.time()
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            probs = torch.nn.functional.softmax(outputs, dim=1)  # Get probability predictions
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())  # Store probabilities
            
            iter_costs.append(loss.item())  # Store loss
            iter_elapsed_time = time.time() - iter_start_time
            iter_times.append(iter_elapsed_time)  # Store time
        
        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        
        epoch_loss = running_loss / total
        accuracy = 100 * correct / total
        
        # Convert to numpy arrays if needed
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # Fix the compute_f1_auc call by passing all required parameters
        try:
            # Try with the full signature including y_prob
            f1_result = compute_f1_auc(all_targets, all_preds, all_probs)
            
            # Handle the case where f1_result might be a tuple of (f1, auc)
            if isinstance(f1_result, tuple):
                f1_value = f1_result[0]  # Extract just the F1 score
            else:
                f1_value = f1_result
                
        except Exception as e:
            # Fallback: if function has changed or isn't compatible, calculate F1 directly
            f1_value = f1_score(all_targets, all_preds, average='macro')
            print(f"Warning: Using direct F1 calculation instead of compute_f1_auc. Error: {e}")
        
        # Compute gradient norms
        if layer_names:
            gradient_norms = compute_gradient_norms(model, layer_names)
            for layer in layer_names:
                if layer in gradient_norms and gradient_norms[layer]:
                    gradient_norms_history[layer].extend(gradient_norms[layer])  # Append norms
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()
        
        metrics['epoch'].append(epoch)
        metrics['loss'].append(epoch_loss)
        metrics['accuracy'].append(accuracy)
        metrics['f1_score'].append(f1_value)
        
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1_value:.4f}")
    
    total_time = time.time() - training_start
    cumulative_times = list(np.cumsum(iter_times)) # changed from epoch_times to iter_times
    print(f"Total training time: {total_time:.2f} seconds")
    return metrics, cumulative_times, gradient_norms_history, iter_costs # added iter_costs
