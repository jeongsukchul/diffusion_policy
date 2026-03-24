from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class NullImageRunner(BaseImageRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir=output_dir)

    def run(self, policy):
        del policy
        return {}
