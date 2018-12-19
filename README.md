# VAT-pytorch
Virtual Adversarial Training (VAT) implementation for Pytorch

* Distributional Smoothing with Virtual Adversarial Training - https://arxiv.org/abs/1507.00677
* Virtual Adversarial Training: a Regularization Method for Supervised and Semi-supervised Learning - 
https://arxiv.org/abs/1704.03976

## Usage
```
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
    cross_entropy = nn.CrossEntropyLoss()

    # LDS should be calculated before the forward for cross entropy
    lds = vat_loss(model, data)
    output = model(data)
    loss = cross_entropy(output, target) + args.alpha * lds
    loss.backward()
    optimizer.step()
```
