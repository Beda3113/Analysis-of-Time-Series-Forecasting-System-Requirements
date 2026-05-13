import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Создаём простую модель
X = np.random.rand(100, 5).reshape(-1, 5)
y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(100) * 0.1

model = LinearRegression()
model.fit(X, y)

# Сохраняем модель
with open('test_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Модель сохранена в test_model.pkl")
