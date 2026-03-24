from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class NullLowdimRunner(BaseLowdimRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir=output_dir)

    def run(self, policy):
        del policy
        return {}
