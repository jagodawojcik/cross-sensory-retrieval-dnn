from evaluation import evaluate
from model import EmbeddingNet, TripletLoss, TripletDataset

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import os


# Create a directory to save your results
RESULTS_DIRECTORY = 'results'
# Number of epochs and margin for triplet loss
EPOCHS = 20001
MARGIN = 0.5


# Set device to gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_with_triplet_loss(epochs=EPOCHS, batch_size=1):
    print("STARTING TRAINING...")

    #Create a directory to save your results
    if os.path.exists(RESULTS_DIRECTORY): 
        raise Exception(f"Directory {RESULTS_DIRECTORY} already exists, please delete it before running this script again.")

    print(f"Directory {RESULTS_DIRECTORY} does not exist, creating...")
    os.makedirs(RESULTS_DIRECTORY)

    # Ask the user for input
    user_input = input("Please enter any identification information about this training: ")
    # Open the file in write mode ('w')
    with open(f"{RESULTS_DIRECTORY}/information.txt", "w") as file:
        # Write the user's input to the file
        file.write(user_input)
        file.write("\n")
        file.write(f"Training with margin {MARGIN} and {EPOCHS} epochs.")

    # Load your embeddings
    audio_embeddings = np.load("audio_embeddings_kaggle_train.npy", allow_pickle=True).item()
    tactile_embeddings = np.load("tactile_embeddings_kaggle_train.npy", allow_pickle=True).item()
    audio_embeddings_test = np.load("audio_embeddings_kaggle_test.npy", allow_pickle=True).item()  
    tactile_embeddings_test = np.load("tactile_embeddings_kaggle_test.npy", allow_pickle=True).item()  

    # Instantiate your dataset and dataloader
    triplet_dataset = TripletDataset(audio_embeddings, tactile_embeddings)
    triplet_dataloader = DataLoader(triplet_dataset, batch_size=1, shuffle=True)

    # Initialize loss function
    triplet_loss = TripletLoss(margin=MARGIN)

    model = EmbeddingNet(embedding_dim=200).to(device)
    # Initialize your optimizer, suppose you have a model named "model"
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Create a directory to save your results
    results_map = {
        'tactile2audio': [],
        'audio2tactile': []
    }
    triplet_loss_save = {
        'triplet_loss': []
    }

    best_map_pairs = {
        'MAP_pairs': []
    }

    # Initialize max MAP values to get best MAP results during training
    max_audio2tactile = 0.0
    max_tactile2audio = 0.0

    # Start training loop
    for epoch in range(EPOCHS):
        total_loss = 0

        for i, (anchor, positive, negative, label) in enumerate(triplet_dataloader):
            # Move to device
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Pass data through the model
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)

            # Compute the loss
            loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
            
            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Add loss
            total_loss += loss.item()

        avg_loss = total_loss / len(triplet_dataloader)
        triplet_loss_save['triplet_loss'].append(avg_loss)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, EPOCHS, avg_loss))

        if epoch % 100 == 0:
            new_audio_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in audio_embeddings.items()}
            new_tactile_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in tactile_embeddings.items()}
            
            with torch.no_grad():
                new_audio_embeddings_test = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in audio_embeddings_test.items()}
                new_tactile_embeddings_test = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in tactile_embeddings_test.items()}

            
            # Evaluate embeddings
            MAP_tactile2audio, MAP_audio2tactile = evaluate(new_audio_embeddings_test, new_tactile_embeddings_test, new_audio_embeddings, new_tactile_embeddings)
            
            if MAP_tactile2audio > max_tactile2audio:
                max_tactile2audio = MAP_tactile2audio
                best_map_pairs['MAP_pairs'].append((epoch, MAP_tactile2audio, MAP_audio2tactile))
                np.save('{}/trained_audio_embeddings_{}.npy'.format(RESULTS_DIRECTORY, epoch), new_audio_embeddings)
                np.save('{}/trained_tactile_embeddings_{}.npy'.format(RESULTS_DIRECTORY, epoch), new_tactile_embeddings)
                torch.save(model.state_dict(), f"{RESULTS_DIRECTORY}/model_best_tactile2audio.pth")
                
            if MAP_audio2tactile > max_audio2tactile:
                max_audio2tactile = MAP_audio2tactile
                best_map_pairs['MAP_pairs'].append((epoch, MAP_tactile2audio, MAP_audio2tactile))
                np.save('{}/trained_audio_embeddings_{}.npy'.format(RESULTS_DIRECTORY, epoch), new_audio_embeddings)
                np.save('{}/trained_tactile_embeddings_{}.npy'.format(RESULTS_DIRECTORY, epoch), new_tactile_embeddings)
                torch.save(model.state_dict(), f"{RESULTS_DIRECTORY}/model_best_audio2tactile.pth")


            # Add the results to the map
            results_map['tactile2audio'].append(MAP_tactile2audio)
            results_map['audio2tactile'].append(MAP_audio2tactile)

    # Save the map results as a JSON file
    with open('{}/map_results_{}.json'.format(RESULTS_DIRECTORY, epoch), 'w') as f:
        json.dump(results_map, f)
    with open('{}/triplet_loss.json'.format(RESULTS_DIRECTORY), 'w') as f:
        json.dump(triplet_loss_save, f)
    with open('{}/best_map_pairs.json'.format(RESULTS_DIRECTORY), 'w') as f:
        json.dump(best_map_pairs, f)

    # Plot the results
    plt.figure(figsize=(12,6))
    plt.plot(range(len(results_map['tactile2audio'])), results_map['tactile2audio'], label='Tactile to Audio')
    plt.plot(range(len(results_map['audio2tactile'])), results_map['audio2tactile'], label='Audio to Tactile')
    plt.xlabel('Triplet Loss Training Epoch')
    plt.ylabel('MAP')
    plt.legend()
    plt.title('MAP Results - Triplet Loss Training')
    plt.savefig('{}/map_plot_{}.png'.format(RESULTS_DIRECTORY, epoch))
    plt.close()

    #Print best results and save them to an information file
    print('MAP Tactile to Audio: {}'.format(max_tactile2audio))
    print('MAP Audio to Tactile: {}'.format(max_audio2tactile))

    with open(f"{RESULTS_DIRECTORY}/information.txt", "a") as file:
        # Write the user's input to the file
        file.write(f"\nMAP Tactile to Audio: {max_tactile2audio}")
        file.write(f"\nMAP Audio to Tactile: {max_audio2tactile}")

    # Plot the triplet loss
    plt.figure(figsize=(12,6))
    plt.plot(range(len(triplet_loss_save['triplet_loss'])), triplet_loss_save['triplet_loss'], label='Triplet Loss')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Triplet Loss', fontsize=18)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Triplet Loss Training', fontsize=18)
    plt.savefig(f'{RESULTS_DIRECTORY}/triplet_loss_plot.png')
    plt.close()


if __name__ == '__main__':
    train_with_triplet_loss()


