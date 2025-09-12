import torch.nn as nn
import torch


class Critic(nn.Module):
    """
    Critic class for estimating the state-action value function.
    Can support multiple (ensemble) critics.
    Can support goal conditioning.
    """

    def __init__(
        self,
        perceptual_features: int,
        action_features: int,
        latent_goal_features: int,
        hidden_features: int,
        num_layers: int,
        num_critics: int,
        activation: str,
    ):
        super(Critic, self).__init__()

        self.critics = nn.ModuleList()
        input_features = (
            perceptual_features + action_features + latent_goal_features
            if latent_goal_features > 0
            else perceptual_features + action_features
        )
        for _ in range(num_critics):
            layers = []
            for i in range(num_layers):
                in_features = input_features if i == 0 else hidden_features
                out_features = 1 if i == num_layers - 1 else hidden_features
                layers.append(nn.Linear(in_features, out_features))
                if i < num_layers - 1:
                    layers.append(getattr(nn, activation)())
            self.critics.append(nn.Sequential(*layers))

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor = None, reduce_type: str = None
    ) -> torch.Tensor:
        x = torch.cat([state, action, goal], dim=-1) if goal is not None else torch.cat([state, action], dim=-1)
        if reduce_type == "first":
            return self.critics[0](x)
        elif reduce_type == "min":
            return torch.min(torch.stack([critic(x) for critic in self.critics]), dim=0).values
        elif reduce_type is None:
            return torch.stack([critic(x) for critic in self.critics])
        else:
            raise ValueError(f"Unknown reduce_type: {reduce_type}")
