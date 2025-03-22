import torch

def generate(model, idx, max_new_tokens, context_size, 
             temperature=0.0, top_k=None, eos_idx=None):
    """
    Generate text from the model
    """
    # Generate the text one token at a time
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] # (batch_size, vocab_size)

        if top_k is not None:
            # Get the top k logits
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val[:, None],
                torch.tensor(-float("Inf")).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature # Apply temperature scaling
            probs = torch.softmax(logits, dim=-1)
            # Sample from the distribution to get the next token
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy sampling (no randomness)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_idx:
            # If the next token is the EOS token, stop generation
            break

        # Add the new token to the context
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
        
            
            
            
            
            