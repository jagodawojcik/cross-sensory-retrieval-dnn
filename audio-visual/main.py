from train_c_entropy import train_with_cross_entropy
from train_triplet_loss import train_with_triplet_loss

EPOCHS_C_ENTROPY = 50
EPOCHS_TRIPLET_LOSS = 40
BATCH_SIZE = 5

def main():
    print("---------Starting Cross Entropy Training-----------")
    train_with_cross_entropy(EPOCHS_C_ENTROPY, BATCH_SIZE)
    print("-----------Cross Entropy Training Completed-----------")

    # print("----------Starting Triplet Loss Training-----------")
    # train_with_triplet_loss(EPOCHS_TRIPLET_LOSS, BATCH_SIZE)
    # print("----------Triplet Loss Training Completed-----------")

    
if __name__ == "__main__":
    main()