from copy import deepcopy
from configs.abiomed.default import default_args


abiomed_args = deepcopy(default_args)
abiomed_args["rollout_length"] = 5
abiomed_args["penalty_coef"] = 1.0
abiomed_args["real_ratio"] = 0.5