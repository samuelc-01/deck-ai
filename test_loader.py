from imageFolder import train_loader, val_loader

print("Número de batches de treino:", len(train_loader))
print("Número de batches de validação:", len(val_loader))

imgs, labels = next(iter(train_loader))
print("Tamanho do batch:", imgs.shape)
print("Labels do batch:", labels)
