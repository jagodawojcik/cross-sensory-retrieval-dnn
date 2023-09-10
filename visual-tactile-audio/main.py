from train_c_entropy import train_with_cross_entropy
from train_triplet_loss import train_with_triplet_loss

EPOCHS_C_ENTROPY = 50
EPOCHS_TRIPLET_LOSS = 40
BATCH_SIZE = 5
EPOCHS_PRETRAIN = 15

def main():
    print("---------Starting Cross Entropy Training-----------")
    train_with_cross_entropy(dominating_mod="audio", epochs_pre=EPOCHS_PRETRAIN, epochs_cross_entropy=EPOCHS_C_ENTROPY, batch_size=BATCH_SIZE)
    print("-----------Cross Entropy Training Completed-----------")
    train_with_cross_entropy(dominating_mod="visual", epochs_pre=EPOCHS_PRETRAIN, epochs_cross_entropy=EPOCHS_C_ENTROPY, batch_size=BATCH_SIZE)
    train_with_cross_entropy(dominating_mod="tactile", epochs_pre=EPOCHS_PRETRAIN, epochs_cross_entropy=EPOCHS_C_ENTROPY, batch_size=BATCH_SIZE)
    # print("----------Starting Triplet Loss Training and Evaluation-----------")
    # train_with_triplet_loss(epochs=EPOCHS_TRIPLET_LOSS)
    # print("----------Triplet Loss Training Completed-----------")


if __name__ == "__main__":
    main()
