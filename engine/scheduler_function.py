def scheduler_exp(total_steps, power=0.9):
    return lambda step: (1 - step / total_steps) ** power