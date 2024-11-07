import math
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import Image, ImageDraw, ImageFont

# Carregando as matrizes suavizadas
matrizes_suavizadas = np.load('matrizes_suavizadas.npy')

# Total de matrizes carregadas
print(f"Total de matrizes suavizadas: {len(matrizes_suavizadas)}")

# Função de ativação sigmoidg
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Inicialização dos pesos com o método Xavier


def initialize_weights(input_size, hidden_size, output_size):
    w1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
    b1 = np.zeros((1, hidden_size))
    w2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
    b2 = np.zeros((1, output_size))
    return w1, b1, w2, b2

# Função de feedforward
def feedforward(X, w1, b1, w2, b2):
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)  # Usa sigmoid na camada de saída
    return z1, a1, z2, a2

# Função de perda
def loss(out, Y):
    # Calcula a soma dos erros quadráticos
    squared_errors = np.square(out - Y)
    total_error = np.sum(squared_errors)
    # Calcula a média
    return 0.5 * total_error / len(Y)

# Função de backpropagation
def backpropagation(X, y_true, z1, a1, a2, w2):
    m = X.shape[0]

    # Gradiente da camada de saída
    dz2 = a2 - y_true
    dw2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    # Gradiente da camada oculta
    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * a1 * (1 - a1)
    dw1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dw1, db1, dw2, db2

# Função de treinamento


def train(X, Y, w1, b1, w2, b2, learning_rate, epochs):
    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        # Feedforward
        z1, a1, z2, a2 = feedforward(X, w1, b1, w2, b2)

        # Cálculo da perda
        current_loss = loss(a2, Y)
        loss_history.append(current_loss)

        # Cálculo da acurácia
        predictions = np.round(a2)  # Round para valores binários
        accuracy = np.mean(predictions == Y) * 100
        acc_history.append(accuracy)

        # Backpropagation
        dw1, db1, dw2, db2 = backpropagation(X, Y, z1, a1, a2, w2)

        # Atualização dos pesos e bias
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2

        # Impressão periódica do progresso
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {
                  epoch + 1}/{epochs} - Loss: {current_loss:.4f} - Accuracy: {accuracy:.2f}%")

    return w1, b1, w2, b2, loss_history, acc_history

# Função para converter labels para one-hot encoding


def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]


# Preparação dos dados
X = matrizes_suavizadas.reshape(len(matrizes_suavizadas), -1)
input_size = X.shape[1]

# Supondo 1.000 exemplos para cada número de 0 a 9
y_train = np.tile(np.arange(10), 1000)
Y = y_train[:len(X)]
Y = one_hot_encode(Y, num_classes=10)

# Normalização dos dados
X = X / 255.0

# Divisão dos dados para treino e teste
num_classes = 10  # Para números de 0 a 9
images_per_class = 1000  # Supondo 1000 imagens para cada classe
train_split = 0.7  # 70% para treino

X_train, Y_train, X_test = [], [], []

for i in range(num_classes):
    # Criando o índice correto para cada número
    start_idx = i * images_per_class
    end_idx = start_idx + images_per_class

    images_for_class = X[start_idx:end_idx]
    labels_for_class = Y[start_idx:end_idx]

    # Dividir para treino e teste
    split_idx = int(images_per_class * train_split)

    # Adicionar imagens de treino e seus rótulos
    X_train.append(images_for_class[:split_idx])
    Y_train.append(labels_for_class[:split_idx])

    # Adicionar imagens de teste sem rótulos
    X_test.append(images_for_class[split_idx:])

# Concatenar as listas de imagens corretamente
X_train = np.concatenate(X_train)
Y_train = np.concatenate(Y_train)
X_test = np.concatenate(X_test)

# Verificar o formato das variáveis
print(f'Formato de X_train: {X_train.shape}')
print(f'Formato de Y_train: {Y_train.shape}')
print(f'Formato de X_test: {X_test.shape}')

# Inicializar pesos
hidden_size = 256
output_size = 10
w1, b1, w2, b2 = initialize_weights(input_size, hidden_size, output_size)

# Treinar a rede
learning_rate = 0.01
epochs = 1000

w1, b1, w2, b2, loss_history, acc_history = train(
    X_train, Y_train, w1, b1, w2, b2, learning_rate, epochs)

# Criando a figura e subplots lado a lado
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# Acurácia
axs[0].plot(acc_history, label='Acurácia de Treinamento')
axs[0].set_xlabel('Épocas')
axs[0].set_ylabel('Acurácia (%)')
axs[0].set_title('Acurácia durante o Treinamento')
axs[0].legend()
# Perda
axs[1].plot(loss_history, label='Perda de Treinamento')
axs[1].set_xlabel('Épocas')
axs[1].set_ylabel('Perda')
axs[1].set_title('Perda durante o Treinamento')
axs[1].legend()
# Exibindo os gráficos
plt.tight_layout()
plt.show()


# teste

# Função para criar uma imagem 30x30 com um número aleatório de 0 a 9
def create_random_digit_image(size=(30, 30)):
    # Criar uma imagem em branco
    image = Image.new('L', size, 255)  # 'L' para imagem em escala de cinza
    draw = ImageDraw.Draw(image)

    # Escolher um número aleatório de 0 a 9
    number = random.randint(0, 9)

    # Usar uma fonte simples para desenhar o número (pode ser necessário ajustar o caminho da fonte)
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Fonte Arial, tamanho 20
    except IOError:
        # Caso a fonte não esteja disponível, usa a fonte padrão
        font = ImageFont.load_default()

    # Calcular a posição do número para centralizar na imagem
    text = str(number)
    # Calcula a caixa delimitadora do texto
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)

    # Desenhar o número na imagem
    draw.text(position, text, font=font, fill=0)  # '0' para cor preta

    return np.array(image), number  # Retorna a imagem e o número gerado


# Criar uma imagem com um número aleatório de 0 a 9
image, number = create_random_digit_image()

# Exibir a imagem criada
plt.imshow(image, cmap='gray')
plt.title(f"Número Aleatório: {number}")
plt.show()

# Preprocessar a imagem
image_flattened = image.flatten()  # Achatar a imagem para formato de vetor
image_normalized = image_flattened / 255.0  # Normalizar para 0-1

# Alimentar a imagem na rede neural para previsão
z1, a1, z2, a2 = feedforward(image_normalized.reshape(1, -1), w1, b1, w2, b2)

# Fazer a previsão
prediction = np.argmax(a2)  # Prever a classe com maior valor de saída
print(f"Previsão da rede: {prediction}, Número verdadeiro: {number}")
