def calculate_ranks(true_entity_score, all_scores):
    """
    Calculates the rank of the true entity.

    Parameter
    ---------
    :param true_entity_scores: torch.Tensor - shape(batch, 1)
        The score of the true entity
    :param all_scores: torch.Tensor - shape(batch, num_entities)
        The scores of all entities

    :return: float
        The rank of the true entity
    """
    assert len(true_entity_score.shape) == 2
    assert len(all_scores.shape) == 2

    all_scores = all_scores > true_entity_score
    true_rank = all_scores.sum(dim=1) + 1

    return true_rank
