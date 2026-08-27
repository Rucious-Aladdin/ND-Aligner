def get_linear_anneal_weight(
    current_step: int,
    start_step: int,
    end_step: int,
    initial_weight: float,
    final_weight: float,
) -> float:
    """
    Calculates weight using piecewise linear annealing.
    - step <= start_step: initial_weight
    - start_step < step < end_step: linear interpolation
    - step >= end_step: final_weight
    """
    if current_step <= start_step:
        return initial_weight
    
    if current_step >= end_step:
        return final_weight
    
    # Linear interpolation
    progress = (current_step - start_step) / (end_step - start_step)
    weight = initial_weight + progress * (final_weight - initial_weight)
    return weight
