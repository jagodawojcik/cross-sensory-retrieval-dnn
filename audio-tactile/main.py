from train_c_entropy import train_with_cross_entropy
from train_triplet_loss import train_with_triplet_loss
from train_c_entropy_pretrain import train_c_entropy_pretrain

EPOCHS_C_ENTROPY = 50
EPOCHS_PRETRAIN = 15
EPOCHS_TRIPLET_LOSS = 40
BATCH_SIZE = 5

def main():
    print("---------Starting Cross Entropy Training-----------")
    train_with_cross_entropy(EPOCHS_C_ENTROPY, BATCH_SIZE)
    print("-----------Cross Entropy Training Completed-----------")

    # print("----------Starting Triplet Loss Training and Evaluation-----------")
    # train_with_triplet_loss(epochs=EPOCHS_TRIPLET_LOSS)
    # print("----------Triplet Loss Training Completed-----------")

    print("----------Starting Cross Entropy with Pretraining-----------")
    train_c_entropy_pretrain(EPOCHS_PRETRAIN, EPOCHS_C_ENTROPY, BATCH_SIZE)
    print("----------Cross Entropy with Pretraining Completed-----------")


if __name__ == "__main__":
    main()