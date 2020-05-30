# Task4 模型训练与验证

#### 构建验证集

```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=10, 
    shuffle=True, 
    num_workers=10, 
)
    
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=10, 
    shuffle=False, 
    num_workers=10, 
)

model = SVHN_Model1()
criterion = nn.CrossEntropyLoss (size_average=False)
optimizer = torch.optim.Adam(model.parameters(), 0.001)
best_loss = 1000.0
for epoch in range(20):
    print('Epoch: ', epoch)

    train(train_loader, model, criterion, optimizer, epoch)
    val_loss = validate(val_loader, model, criterion)
    
    # 记录下验证集精度
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), './model.pt')
```

Epoch训练

```python
def train(train_loader, model, criterion, optimizer, epoch):
    # 切换模型为训练模式
    model.train()

    for i, (input, target) in enumerate(train_loader):
        c0, c1, c2, c3, c4, c5 = model(data[0])
        loss = criterion(c0, data[1][:, 0]) + \
                criterion(c1, data[1][:, 1]) + \
                criterion(c2, data[1][:, 2]) + \
                criterion(c3, data[1][:, 3]) + \
                criterion(c4, data[1][:, 4]) + \
                criterion(c5, data[1][:, 5])
        loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Epoch验证

```python
def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            c0, c1, c2, c3, c4, c5 = model(data[0])
            loss = criterion(c0, data[1][:, 0]) + \
                    criterion(c1, data[1][:, 1]) + \
                    criterion(c2, data[1][:, 2]) + \
                    criterion(c3, data[1][:, 3]) + \
                    criterion(c4, data[1][:, 4]) + \
                    criterion(c5, data[1][:, 5])
            loss /= 6
            val_loss.append(loss.item())
    return np.mean(val_loss)
```

#### 模型保存加载

```python
torch.save(model_object.state_dict(), 'model.pt') 
```

```pyhton 
model.load_state_dict(torch.load(' model.pt')) 
```

