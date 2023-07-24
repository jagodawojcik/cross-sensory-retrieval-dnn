import numpy as np

RANK_K = 5

def prepare_data_for_evaluation(embeddings):
    labels = []
    embeddings_list = []
    for label, embedding in embeddings.items():
        labels.extend([label]*len(embedding))
        embeddings_list.extend(embedding)
    return np.array(embeddings_list), np.array(labels)

def average_precision(relevant_scores, k):
    if len(relevant_scores) == 0:
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i, relevant in enumerate(relevant_scores[:k]):
        if relevant > 0:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(relevant_scores), k)

def calculate_MAP(queries, retrievals, query_labels, retrieval_labels):
    APs = []
    for i in range(len(queries)):
        query = queries[i]
        current_query_label = query_labels[i]

        # calculate the distance between the query and all retrievals
        distances = np.linalg.norm(retrievals - query, axis=1)

        # sort the distances to find the closest and get their indices
        closest_indices = np.argsort(distances)

        # get the labels for the closest instances
        closest_labels = retrieval_labels[closest_indices]

        # determine which retrievals are true positives
        relevant_retrievals = (closest_labels == current_query_label)

        # calculate the average precision for this query
        ap = average_precision(relevant_retrievals, RANK_K)
        APs.append(ap)

    # the mean average precision is the mean of the average precision values for all queries
    return np.mean(APs)


def evaluate(audio_test_embeddings, tactile_test_embeddings, audio_train_embeddings, tactile_train_embeddings):
    audio_test_queries, audio_test_query_labels = prepare_data_for_evaluation(audio_test_embeddings)
    tactile_test_queries, tactile_test_query_labels = prepare_data_for_evaluation(tactile_test_embeddings)
    audio_train_queries, audio_train_query_labels = prepare_data_for_evaluation(audio_train_embeddings)
    tactile_train_queries, tactile_train_query_labels = prepare_data_for_evaluation(tactile_train_embeddings)

    # Calculate MAP for tactile2audio retrieval
    MAP_tactile2audio = calculate_MAP(tactile_test_queries, audio_train_queries, tactile_test_query_labels, audio_train_query_labels)

    # Calculate MAP for audio2tactile retrieval
    MAP_audio2tactile = calculate_MAP(audio_test_queries, tactile_train_queries, audio_test_query_labels, tactile_train_query_labels)

    return MAP_tactile2audio, MAP_audio2tactile


if __name__ == "__main__":
    evaluate()
