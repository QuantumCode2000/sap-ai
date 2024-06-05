


import time
import random
import csv
from colorama import Fore, Back, Style
import matplotlib.pyplot as plt

def training_results(num_epochs, training_time_per_epoch, wait_time, initial_accuracy, final_accuracy):

    # Definición de variables
    progress = 0  # Progreso actual del entrenamiento
    epoch_start_time = 0  # Tiempo de inicio de la época actual
    total_training_time = 0  # Tiempo total de entrenamiento
    accuracy_increment = (final_accuracy - initial_accuracy) / num_epochs

    accuracies_train = []
    accuracies_val = []
    losses_train = []
    losses_val = []

    # Simulación del entrenamiento
    with open("training_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Train Accuracy", "Validation Accuracy", "Train Loss", "Validation Loss", "Training Time (s)"])

        for epoch in range(num_epochs):
            # Registrar el tiempo de inicio de la época
            epoch_start_time = time.time()

            # Imprimir información de la época
            print(f"\n**Epoch {epoch + 1} of {num_epochs}**")

            # Simular entrenamiento dentro de la época
            for i in range(200):
                # Actualizar el progreso
                progress = i / 200

                # Simular la computación
                time.sleep(random.uniform(0.01, 0.05))

                # Imprimir la barra de progreso dinámica
                print(f"\r{Fore.CYAN}[{'-' * int(40 * (1 - progress))}{'=' * int(40 * progress)}]{Fore.RESET} {int(progress * 100)}%", end="")

                # **Ralentizar la simulación dentro de la época**
                for j in range(10):
                    time.sleep(0.01)

            # Calcular el tiempo de entrenamiento de la época
            epoch_training_time = time.time() - epoch_start_time
            total_training_time += epoch_training_time

            # Generar precisión y pérdida aleatorias
            train_accuracy = initial_accuracy + epoch * accuracy_increment + random.uniform(-0.05, 0.05)
            val_accuracy = initial_accuracy + epoch * accuracy_increment + random.uniform(-0.05, 0.05)
            train_loss = random.uniform(0.3, 0.5) - (epoch * 0.01)
            val_loss = random.uniform(0.3, 0.5) - (epoch * 0.01)

            # Asegurar que las precisiones y pérdidas estén dentro de límites razonables
            train_accuracy = max(0, min(1, train_accuracy))
            val_accuracy = max(0, min(1, val_accuracy))
            train_loss = max(0, train_loss)
            val_loss = max(0, val_loss)

            # Imprimir información específica de la época después de la barra de progreso
            print(f"\n{Fore.GREEN}Train accuracy: {train_accuracy:.2%}, Validation accuracy: {val_accuracy:.2%}{Fore.RESET}")
            print(f"{Fore.RED}Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}{Fore.RESET}")

            # Escribir resultados en el archivo .csv
            writer.writerow([epoch + 1, train_accuracy, val_accuracy, train_loss, val_loss, epoch_training_time])

            # Guardar los datos para las gráficas
            accuracies_train.append(train_accuracy)
            accuracies_val.append(val_accuracy)
            losses_train.append(train_loss)
            losses_val.append(val_loss)

            # Esperar entre épocas
            time.sleep(wait_time)

    # Simulación de la finalización del entrenamiento y la exportación del modelo
    print(f"\n{Fore.YELLOW}**Total training time: {total_training_time:.2f} seconds**{Fore.RESET}")
    print(f"{Fore.MAGENTA}**Model saved to 'trained_model.txt'**{Fore.RESET}")  # Simulated model saving

    # Simulación del tiempo de exportación
    export_time = 10  # Tiempo de exportación simulado
    print(f"{Fore.CYAN}**Export time: {export_time:.2f} seconds**{Fore.RESET}")
    print(f"{Fore.GREEN}**Model exported with an accuracy of 92.711232%**{Fore.RESET}")

    # Imprimir mensaje de finalización del entrenamiento
    print("\n**Training completed!**")

    # Graficar los resultados
    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(12, 6))

    # Gráfica de precisión
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracies_train, label='Train Accuracy')
    plt.plot(epochs, accuracies_val, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Gráfica de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs, losses_train, label='Train Loss')
    plt.plot(epochs, losses_val, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

# Llamar a la función
training_results(200, 2, 1, 0.7, 0.9251)


