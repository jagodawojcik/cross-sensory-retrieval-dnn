import torch
import numpy as np
from collections import defaultdict
import os

from logger import logger
from load_data import get_test_loader  
from model import CrossSensoryNetwork, EmbeddingNet
from evaluation import evaluate_audio_tact, evaluate_audio_vis, evaluate_tact_vis

def test_set_performance_evaluate(dominating_modality):

    logger.log("Start final performance evaluation on a test set.")
    
    # Initialize results directory
    FINAL_EVALUATION_RESULTS_DIRECTORY = os.path.join(f"dom-{dominating_modality.value}",f"performance-evaluation-results")

    #Create a directory to save your results
    if os.path.exists(FINAL_EVALUATION_RESULTS_DIRECTORY): 
        raise Exception(f"Directory {FINAL_EVALUATION_RESULTS_DIRECTORY} already exists, please check for existing results.")
    
    logger.log(f"Directory {FINAL_EVALUATION_RESULTS_DIRECTORY} does not exist, creating...")
    os.makedirs(FINAL_EVALUATION_RESULTS_DIRECTORY)

    C_ENTROPY_RESULTS_DIRECTORY = os.path.join(f"dom-{dominating_modality.value}",f"c-entropy-results")
    TRIPLET_RESULTS_DIRECTORY = os.path.join(f"dom-{dominating_modality.value}",f"triplet-loss")

    # Load saved model paths 
    saved_c_entropy_model_path = f"{C_ENTROPY_RESULTS_DIRECTORY}/c-entropy-model.pth"
    saved_triplet_model_path_query_fused = f"{TRIPLET_RESULTS_DIRECTORY}/triplet_model.pth"

    # Device to cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CrossSensoryNetwork().to(device)
    model.load_state_dict(torch.load(saved_c_entropy_model_path))
    model.eval()

    # Initialize test dataloader
    batch_size = 5  
    dataloader = get_test_loader(batch_size)
    test_loader = dataloader['test']

    # Dictionaries to save embeddings
    audio_embeddings_test = defaultdict(list)
    tactile_embeddings_test = defaultdict(list)
    visual_embeddings_test = defaultdict(list)

    # Inference loop
    with torch.no_grad():
        for i, (audio_input, tactile_input, visual_input, targets) in enumerate(test_loader):
            audio_input, tactile_input, visual_input, targets = audio_input.to(device), tactile_input.to(device), visual_input.to(device), targets.to(device)

            # Get outputs and embeddings
            audio_output, tactile_output, visual_output, _ = model(audio_input, tactile_input, visual_input)

            for j in range(audio_output.shape[0]):
                label = targets[j].item()
                audio_embeddings_test[label].append(audio_output[j].cpu().numpy())
                tactile_embeddings_test[label].append(tactile_output[j].cpu().numpy())
                visual_embeddings_test[label].append(visual_output[j].cpu().numpy())

    # Save the intermediate state of embeddings
    np.save(os.path.join(f"{FINAL_EVALUATION_RESULTS_DIRECTORY}","audio_embeddings_test_c_entropy.npy"), dict(audio_embeddings_test))
    np.save(os.path.join(f"{FINAL_EVALUATION_RESULTS_DIRECTORY}","tactile_embeddings_test_c_entropy.npy"), dict(tactile_embeddings_test))
    np.save(os.path.join(f"{FINAL_EVALUATION_RESULTS_DIRECTORY}","visual_embeddings_test_c_entropy.npy"), dict(visual_embeddings_test))
   
    # Initialize triplet model
    model = EmbeddingNet(embedding_dim=200).to(device)
    model.load_state_dict(torch.load(saved_triplet_model_path_query_fused))
    model.eval()

    # Produce final query embeddings from test set
    tactile_embeddings = np.load(os.path.join(FINAL_EVALUATION_RESULTS_DIRECTORY, "tactile_embeddings_test_c_entropy.npy"), allow_pickle=True).item()
    visual_embeddings = np.load(os.path.join(FINAL_EVALUATION_RESULTS_DIRECTORY, "visual_embeddings_test_c_entropy.npy"), allow_pickle=True).item()
    audio_embeddings = np.load(os.path.join(FINAL_EVALUATION_RESULTS_DIRECTORY, "audio_embeddings_test_c_entropy.npy"), allow_pickle=True).item()
    
    with torch.no_grad():
        new_tactile_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in tactile_embeddings.items()}
        new_visual_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in visual_embeddings.items()}
        new_audio_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in audio_embeddings.items()}
        
    # Load retrieval space embeddings
    retrieval_tactile_embeddings = np.load(os.path.join(C_ENTROPY_RESULTS_DIRECTORY, "tactile_embeddings_train.npy"), allow_pickle=True).item()
    retrieval_visual_embeddings = np.load(os.path.join(C_ENTROPY_RESULTS_DIRECTORY, "visual_embeddings_train.npy"), allow_pickle=True).item()
    retrieval_audio_embeddings = np.load(os.path.join(C_ENTROPY_RESULTS_DIRECTORY, "audio_embeddings_train.npy"), allow_pickle=True).item()

    with torch.no_grad():
        new_retrieval_tactile_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in retrieval_tactile_embeddings.items()}
        new_retrieval_visual_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in retrieval_visual_embeddings.items()}
        new_retrieval_audio_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in retrieval_audio_embeddings.items()}
        
    # Evaluate MAP score
    MAP_tactile2visual, MAP_visual2tactile = evaluate_tact_vis(new_visual_embeddings, new_tactile_embeddings, new_retrieval_visual_embeddings, new_tactile_embeddings)
    MAP_visual2audio, MAP_audio2visual = evaluate_audio_vis(new_audio_embeddings, new_visual_embeddings, new_retrieval_audio_embeddings, new_retrieval_visual_embeddings)
    MAP_tactile2audio, MAP_audio2tactile = evaluate_audio_tact(new_audio_embeddings, new_tactile_embeddings, new_retrieval_audio_embeddings, new_retrieval_tactile_embeddings)

    logger.log("Finished evaluation of final performance.")
    logger.log(f"\nMAP Tactile to Visual: {MAP_tactile2visual}")
    logger.log(f"\nMAP Visual to Tactile: {MAP_visual2tactile}")
    logger.log(f"\nMAP Tactile to Audio: {MAP_tactile2audio}")
    logger.log(f"\nMAP Audio to Tactile: {MAP_audio2tactile}")
    logger.log(f"\nMAP Visual to Audio: {MAP_visual2audio}")
    logger.log(f"\nMAP Audio to Visual: {MAP_audio2visual}")

    RESULTS_DIRECTORY = f'dom-{dominating_modality.value}'

    with open(f"{RESULTS_DIRECTORY}/result.txt", "w") as file:
        file.write(f"\nMAP Tactile to Visual: {MAP_tactile2visual}")
        file.write(f"\nMAP Visual to Tactile: {MAP_visual2tactile}")
        file.write(f"\nMAP Tactile to Audio: {MAP_tactile2audio}")
        file.write(f"\nMAP Audio to Tactile: {MAP_audio2tactile}")
        file.write(f"\nMAP Visual to Audio: {MAP_visual2audio}")
        file.write(f"\nMAP Audio to Visual: {MAP_audio2visual}")