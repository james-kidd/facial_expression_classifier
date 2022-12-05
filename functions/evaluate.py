import matplotlib.pyplot as plt


def eval_model_stats(model, depth, test_ds):
    loss, acc = model.evaluate(test_ds, verbose=2)
    print(f'\n{depth} Model')
    print("Accuracy: {:5.2f}%".format(100 * acc), "\nLoss: {:5.2f}".format(loss))
    
def plot_model_peformance(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def plot_model_peformance_from_df(hist_df, epochs, depth):
    acc = hist_df['accuracy']
    val_acc = hist_df['val_accuracy']

    loss = hist_df['loss']
    val_loss = hist_df['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 4))
    plt.suptitle(f"{depth}", fontsize=12)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()