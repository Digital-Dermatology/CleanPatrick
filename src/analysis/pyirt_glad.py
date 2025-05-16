import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from py_irt.models import abstract_model
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


@abstract_model.IrtModel.register("glad")
class GLADModel(abstract_model.IrtModel):
    def __init__(
        self,
        *,
        priors: str,
        num_items: int,
        num_subjects: int,
        verbose: bool = False,
        experts: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            device=device,
            num_items=num_items,
            num_subjects=num_subjects,
            verbose=verbose,
        )
        if priors not in ["vague", "hierarchical"]:
            raise ValueError("Options for priors are vague and hierarchical")
        self.priors = priors
        self.experts = experts

    def model_vague(self, models, items, obs):
        with pyro.plate("thetas", self.num_subjects, device=self.device):
            ability = pyro.sample(
                "theta",
                dist.Normal(
                    torch.tensor(1.0 if self.experts else 0.0, device=self.device),
                    torch.tensor(1.0, device=self.device),
                ),
            )
        with pyro.plate("bs", self.num_items, device=self.device):
            difficulty = pyro.sample(
                "b",
                dist.Normal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(1.0e3, device=self.device),
                ),
            )
        with pyro.plate("observe_data", obs.size(0), device=self.device):
            pyro.sample(
                "obs",
                dist.Bernoulli(logits=ability[models] * difficulty[items]),
                obs=obs,
            )

    def guide_vague(self, models, items, obs):
        loc_ability = pyro.param(
            "loc_ability", torch.zeros(self.num_subjects, device=self.device)
        )
        scale_ability = pyro.param(
            "scale_ability",
            torch.ones(self.num_subjects, device=self.device),
            constraint=constraints.positive,
        )
        loc_diff = pyro.param(
            "loc_diff", torch.zeros(self.num_items, device=self.device)
        )
        scale_diff = pyro.param(
            "scale_diff",
            torch.empty(self.num_items, device=self.device).fill_(1.0e3),
            constraint=constraints.positive,
        )
        with pyro.plate("thetas", self.num_subjects, device=self.device):
            pyro.sample("theta", dist.Normal(loc_ability, scale_ability))
        with pyro.plate("bs", self.num_items, device=self.device):
            pyro.sample("b", dist.Normal(loc_diff, scale_diff))

    def get_model(self):
        return self.model_vague if self.priors == "vague" else self.model_hierarchical

    def get_guide(self):
        return self.guide_vague if self.priors == "vague" else self.guide_hierarchical

    def fit(self, models, items, responses, num_epochs):
        optim = Adam({"lr": 0.1})
        svi = SVI(self.model_vague, self.guide_vague, optim, loss=Trace_ELBO())
        pyro.clear_param_store()
        for j in range(num_epochs):
            loss = svi.step(models, items, responses)
            if j % 100 == 0 and self.verbose:
                print(f"[epoch {j+1:04d}] loss: {loss:.4f}")
        print(f"[epoch {num_epochs}] loss: {loss:.4f}")

    def export(self):
        return {
            "ability": pyro.param("loc_ability").data.tolist(),
            "difficulty": pyro.param("loc_diff").data.tolist(),
        }

    def predict(self, subjects, items):
        model_params = self.export()
        abilities = np.array([model_params["ability"][i] for i in subjects])
        difficulties = np.array([model_params["difficulty"][i] for i in items])
        return 1 / (1 + np.exp(-abilities * difficulties))


if __name__ == "__main__":
    # Validate (final proba. should be close to the `response`)
    model = GLADModel(priors="vague", num_items=3, num_subjects=3, verbose=True)
    model.fit(
        models=np.array([0, 1, 2]),
        items=np.array([0, 1, 2]),
        responses=torch.tensor([1, 0, 1], dtype=torch.float32),
        num_epochs=5_000,
    )
    print("Predicted Probabilities:", model.predict([0, 1, 2], [0, 1, 2]))
