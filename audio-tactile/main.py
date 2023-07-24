from train_c_entropy import train_with_cross_entropy
from train_triplet_loss import train_with_triplet_loss

EPOCHS_C_ENTROPY = 15
EPOCHS_TRIPLET_LOSS = 40
BATCH_SIZE = 10

def main():
    print("---------Starting Cross Entropy Training-----------")
    train_with_cross_entropy(EPOCHS_C_ENTROPY, BATCH_SIZE)
    print("-----------Cross Entropy Training Completed-----------")

    print("----------Starting Triplet Loss Training and Evaluation-----------")
    train_with_triplet_loss(epochs=EPOCHS_TRIPLET_LOSS)
    print("----------Triplet Loss Training Completed-----------")


if __name__ == "__main__":
    main()